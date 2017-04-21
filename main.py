#!/usr/bin/python2.7

import argparse
import yaml
import sys

import logging.config
from colorlog import ColoredFormatter
from timeit import default_timer as timer
import progressbar as pb
import numpy as np
from nflib import Data, FiringRate, Connectivity
from tools import qifint, TheoreticalComputations, SaveResults, Perturbation, noise, FrequencySpectrum, ColorPlot, \
    LinearStability

import Gnuplot

# Use this option to turn off fifo if you get warnings like:
# line 0: warning: Skipping unreadable file "/tmp/tmpakexra.gnuplot/fifo"
Gnuplot.GnuplotOpts.prefer_fifo_data = 0

__author__ = 'jm'


# Empty class to manage external parameters
# noinspection PyClassHasNoInit
class Options:
    pass


options = None
ops = Options()
pi = np.pi
pi2 = np.pi * np.pi

# We first try to parse optional configuration files:
fparser = argparse.ArgumentParser(add_help=False)
fparser.add_argument('-f', '--file', default="conf.txt", dest='-f', metavar='<file>')
fparser.add_argument('-db', '--debug', default="INFO", dest='db', metavar='<debug>',
                     choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
farg = fparser.parse_known_args()
conffile = vars(farg[0])['-f']
# ####### Debugging #########
debug = getattr(logging, vars(farg[0])['db'].upper(), None)
if not isinstance(debug, int):
    raise ValueError('Invalid log level: %s' % vars(farg[0])['db'])

logformat = "%(log_color)s[%(levelname)-7.8s]%(reset)s %(name)-12.12s:%(funcName)-8.8s: " \
            "%(log_color)s%(message)s%(reset)s"
formatter = ColoredFormatter(logformat, log_colors={
    'DEBUG': 'cyan',
    'INFO': 'white',
    'WARNING': 'yellow',
    'ERROR': 'red',
    'CRITICAL': 'red,bg_white',
})

logging.config.dictConfig(yaml.load(file('logging.conf', 'rstored')))
handler = logging.root.handlers[0]
handler.setLevel(debug)
handler.setFormatter(formatter)
logger = logging.getLogger('simulation')

# We open the configuration file to load parameters (not optional)
try:
    options = yaml.load(file(conffile, 'rstored'))
except IOError:
    logger.error("The configuration file '%s' is missing" % conffile)
    exit(-1)
except yaml.YAMLError, exc:
    logger.error("Error in configuration file:", exc)
    exit(-1)

# We load parameters from the dictionary of the conf file and add command line options (2nd parsing)
parser = argparse.ArgumentParser(
    description='Simulator of a network of ensembles of all-to-all QIF neurons.',
    usage='python %s [-O <options>]' % sys.argv[0])
print "\n******************************************************************"
logger.info('Simulator of a network of ensembles of all-to-all QIF neurons.')
for group in options:
    gr = parser.add_argument_group(group)
    for key in options[group]:
        flags = key.split()
        args = options[group]
        if isinstance(args[key]['default'], bool):
            gr.add_argument(*flags, default=args[key]['default'], help=args[key]['description'], dest=flags[0][1:],
                            action='store_true')
        elif isinstance(args[key]['default'], list):
            tipado = type(args[key]['default'][0])
            gr.add_argument(*flags, default=args[key]['default'], help=args[key]['description'], dest=flags[0][1:],
                            metavar=args[key]['name'], type=tipado,
                            choices=args[key]['choices'], nargs='+')
        else:
            gr.add_argument(*flags, default=args[key]['default'], help=args[key]['description'], dest=flags[0][1:],
                            metavar=args[key]['name'], type=type(args[key]['default']),
                            choices=args[key]['choices'])

# We parse command line arguments:
opts = parser.parse_args(farg[1])
args = parser.parse_args(farg[1], namespace=ops)

# ##################################################################################
# 0) PREPARE FOR CALCULATIONS
# 0.1) Load data object:
d = Data(l=args.n, n=args.N, eta0=args.e, j0=args.j, delta=args.d, tfinal=args.T, system=args.s, fp=args.D, dt=args.dt)
d.tau_s_ex = args.tin * 1.0
d.tau_s_in = args.tex * 1.0
# 0.2) Create connectivity matrix and extract eigenmodes
c = Connectivity(d.l, profile=args.c, fsmodes=args.jk, amplitude=10.0, data=d, degree=args.dg, saved=True)

# 0.3) Load initial conditions
if args.oic is False:
    d.load_ic(0.0, system=d.system)
else:
    # Override initial conditions generator:
    pass
if args.ic:
    logger.info("Forcing initial conditions generation...")
    d.new_ic = True

# 0.4) Load Firing rate class in case qif network is simulated
if d.system != 'nf' and not d.new_ic:
    fr = FiringRate(data=d, swindow=0.5, sampling=0.05)

# 0.5) Set perturbation configuration
# mode patch
if args.m == -1:
    args.m = range(0, 10)

p = list()
p.append(Perturbation(data=d, dt=args.pt, modes=args.m, amplitude=float(args.a), attack=args.A, cntmodes=c.eigenvectors,
                      t0=args.pt0, stype=args.sP, ptype=args.pT, duration=d.total_time))

if args.pV:
    p.amp = 0.0

# 0.6) Define saving paths:
sr = SaveResults(data=d, cnt=c, pert=p, system=d.system, parameters=opts)

# 0.7) Other theoretical tools:
th = TheoreticalComputations(d, c, p)
F = FrequencySpectrum()

# Progress-bar configuration
widgets = ['Progress: ', pb.Percentage(), ' ',
           pb.Bar(marker=pb.RotatingMarker()), ' ', pb.ETA(), ' ']

###################################################################################
# 1) Simulation (Integrate the system)
print('Simulating ...')
pbar = pb.ProgressBar(widgets=widgets, maxval=10 * (d.nsteps + 1)).start()
time1 = timer()
tstep = 0
temps = 0
nois = 0.0
noise_E = noise_I = 0.0
kp = k = 0
freq = 0.0
# Time loop
while temps < d.tfinal:
    # Time step variables
    kp = tstep % d.nsteps
    k = (tstep + d.nsteps - 1) % d.nsteps
    k2p = tstep % 2
    k2 = (tstep + 2 - 1) % 2

    # ######################## - PERTURBATION  - ##
    # Perturbation at certain time
    for pert in p:
        if pert.t0step == tstep:
            pert.pbool = True
        if pert.pbool and not d.new_ic:
            if temps >= pert.t0:
                pert.timeevo(temps, freq=freq)
                pt0step = tstep * 1
        d.it[kp, :] += pert.input
    # Noisy perturbation
    if args.ns and not d.new_ic:
        if tstep % 1 == 0:
            # nois = np.sqrt(2.0 * d.dt / d.tau * args.nD) * np.random.randn(d.l / 10)
            # nois = np.dot(p.auxMat, nois)
            nois = np.sqrt(2.0 * d.dt / d.tau * args.nD) * np.random.rand(d.l)
        else:
            nois = 0.0

    d.it[kp, :] += d.tau / d.dt * nois

    # ######################## -  INTEGRATION  - ##
    # ######################## -      qif      - ##
    if d.system == 'qif' or d.system == 'both':
        # TODO: we have not yet implement synaptic kinetics in the QIF system
        tsyp = tstep % d.T_syn
        tskp = tstep % d.spiketime
        tsk = (tstep + d.spiketime - 1) % d.spiketime
        # We compute the Mean-field vector s_j
        se = (1.0 / d.Ne) * np.dot(c.cnt_ex, np.dot(d.auxMatE, np.dot(d.spikes_e, d.a_tau[:, tsyp])))
        si = (1.0 / d.Ni) * np.dot(c.cnt_in, np.dot(d.auxMatI, np.dot(d.spikes_i, d.a_tau[:, tsyp])))
        s = se + si

        # Another perturbation (directly changing mean potentials)
        for pert in p:
            if tstep == pert.t0step and args.pV and not d.new_ic:
                d.matrixE[:, 0] += np.dot(pert.auxMatD, args.pD * pert.smod)
                d.matrixI[:, 0] += args.sym * np.dot(pert.auxMatD, args.pD * pert.smod)

        if d.fp == 'noise':
            noise_E = np.sqrt(2.0 * d.dt / d.tau * d.delta) * noise(d.Ne)
            noise_I = np.sqrt(2.0 * d.dt / d.tau * d.delta) * noise(d.Ni)

        # Excitatory
        d.matrixE = qifint(d.matrixE, d.matrixE[:, 0], d.matrixE[:, 1], d.etaE + d.tau / d.dt * noise_E,
                           s + d.it[kp, :],
                           temps, d.Ne, d.dNe, d.dt, d.tau, d.vpeak, d.refr_tau, d.tau_peak)
        # Inhibitory
        d.matrixI = qifint(d.matrixI, d.matrixI[:, 0], d.matrixI[:, 1], d.etaI + d.tau / d.dt * noise_I,
                           s + args.sym * d.it[kp, :],
                           temps, d.Ni, d.dNi, d.dt, d.tau, d.vpeak, d.refr_tau, d.tau_peak)

        # Prepare spike matrices for Mean-Field computation and firing rate measure
        # Excitatory
        d.spikes_e_mod[:, tsk] = 1 * d.matrixE[:, 2]  # We store the spikes
        d.spikes_e[:, tsyp] = 1 * d.spikes_e_mod[:, tskp]
        # Inhibitory
        d.spikes_i_mod[:, tsk] = 1 * d.matrixI[:, 2]  # We store the spikes
        d.spikes_i[:, tsyp] = 1 * d.spikes_i_mod[:, tskp]

        # If we are just obtaining the initial conditions (a steady state) we don't need to
        # compute the firing rate.
        if not d.new_ic:
            # Voltage measure:
            # vma = (d.matrix[:, 1] <= temps)  # Neurons which are not in the refractory period
            # fr.vavg0[vma] += d.matrix[vma, 0]
            # fr.vavg += 1

            # ######################## -- FIRING RATE MEASURE -- ##
            fr.frspikes_e[:, tstep % fr.wsteps] = 1 * d.spikes_e[:, tsyp]
            fr.frspikes_i[:, tstep % fr.wsteps] = 1 * d.spikes_i[:, tsyp]
            fr.firingrate(tstep)
            # Distribution of Firing Rates
            if tstep > 0:
                fr.tspikes_e2 += d.matrixE[:, 2]
                fr.tspikes_i2 += d.matrixI[:, 2]
                fr.ravg2 += 1  # Counter for the "instantaneous" distribution
                fr.ravg += 1  # Counter for the "total time average" distribution

                # Check if both populations (ex. and inh.) are doing the same thing
                # if not np.all(d.matrixE == d.matrixI):
                #     logger.debug("'Symmetry' breaking... not exactly.")

    # ######################## -  INTEGRATION  - ##
    # ######################## --   FR EQS.   -- ##
    if d.system == 'nf' or d.system == 'both':
        # Another perturbation (directly changing mean potentials)
        for pert in p:
            if tstep == pert.t0step and args.pV and not d.new_ic:
                d.v_ex[k] += args.pD * pert.smod
                d.v_in[k] += args.sym * args.pD * pert.smod

        # We compute the Mean-field vector S ( 1.0/(2.0*pi)*dx = 1.0/l )
        d.sphi[k2p] = (1.0 / d.l * np.dot(c.cnt_ex, d.r_ex[k]) + 1.0 / d.l * np.dot(c.cnt_in, d.r_in[k]))
        # -- Integration -- #
        d.s_ex[kp] = d.s_ex[k] + d.dt / d.tau_s_ex * (d.sphi[k2p] - d.s_ex[k])
        d.r_ex[kp] = d.r_ex[k] + d.dt * (d.delta / pi + 2.0 * d.r_ex[k] * d.v_ex[k])
        d.v_ex[kp] = d.v_ex[k] + d.dt * (d.v_ex[k] ** 2 + d.eta0 + d.s_ex[kp] - pi2 * d.r_ex[k] ** 2 + d.it[kp])
        d.s_in[kp] = d.s_in[k] + d.dt / d.tau_s_in * (d.sphi[k2p] - d.s_in[k])
        d.r_in[kp] = d.r_in[k] + d.dt * (d.delta / pi + 2.0 * d.r_in[k] * d.v_in[k])
        d.v_in[kp] = d.v_in[k] + d.dt * (
            d.v_in[k] ** 2 + d.eta0 + d.s_in[kp] - pi2 * d.r_in[k] ** 2 + args.sym * d.it[kp])

    # Compute the frequency by hand (for a given node, typically at the center)

    # Time evolution
    pbar.update(10 * tstep + 1)
    temps += d.dt
    tstep += 1

# Finish pbar
pbar.finish()
# Stop the timer
print 'Total time: {}.'.format(timer() - time1)

###################################################################################
# 2) Post-Simulation, saving, plotting, analayzing.
logger.debug("Stationary firing rate (excitatory and inhibitory): %f, %f" % (d.r_ex[kp, 0], d.r_in[kp, 0]))
logger.debug("Stationary mean membrane potential (excitatory and inhibitory): %f, %f" % (d.v_ex[kp, 0], d.v_in[kp, 0]))

# Compute distribution of firing rates of neurons
tstep -= 1
temps -= d.dt
# th.thdist = th.theor_distrb(d.sphi[kp])

# Frequency analysis
envelope = {}
if args.Frq:
    plot = ColorPlot(data=d)
    phi = d.l / 2
    F.analyze(d.r_ex[:, phi] - d.r0, 0.0, d.tfinal, d.faketau, method='all')
    F.analyze(plot.filter(d.r_ex - d.r0)[:, phi], 0.0, d.tfinal, d.faketau, method='all')
    for pert in p:
        if pert.ptype == 'oscillatory' or pert.ptype == 'chirp':
            pop = d.l / 2
            if 'qif' in d.systems:
                envelope['qif'] = LinearStability.envelope2extreme(fr.r, fr.tempsfr, tau=d.faketau)
            if 'nf' in d.systems:
                envelope['nf'] = LinearStability.envelope2extreme((d.r_ex + d.r_in) / 2.0, d.tpoints, tau=d.faketau)

# Save initial conditions
if d.new_ic:
    d.save_ic(temps)
    exit(0)

# Save data to dictionary
if not args.nos:
    # Register data to a dictionary
    if 'qif' in d.systems:
        # Distribution of firing rates over all time
        fr.frqif_e = fr.tspikes_e / (fr.ravg * d.dt) / d.faketau
        fr.frqif_i = fr.tspikes_i / (fr.ravg * d.dt) / d.faketau
        fr.frqif = np.concatenate((fr.frqif_e, fr.frqif_i))

        if 'nf' in d.systems:
            d.register_ts(fr, th)
        else:
            d.register_ts(fr)
    else:
        d.register_ts(th=th)

    sr.create_dict()
    sr.results['perturbation']['It'] = d.it
    for pert in p:
        if pert.ptype == 'oscillatory':
            sr.results['perturbation']['freqs'] = pert.freq
            if args.Frq and pert.ptype in ['']:
                for sys in d.systems:
                    sr.results[sys]['fr']['envelope'] = envelope[sys]
        if pert.ptype == 'chirp':
            sr.results['perturbation']['chirp'] = pert.chirp
            for sys in d.systems:
                sr.results[sys]['fr']['envelope'] = envelope[sys]
    sr.save()

# Save just some data and plot
if args.pl:
    plot = ColorPlot(data=d, tfinal=d.tfinal)
    plot.cplot(d.r_ex)
    plot.cplot(d.r_in)

# Preliminary plotting with gnuplot
if args.gpl:
    gpllog = logger.getChild('gnuplot')
    if d.nsteps > 10E6:
        points = d.nsteps / 10E6
        if points <= 1:
            points = 10
    else:
        points = 1
    gpllog.info("Plotting every %d points", points)
    gp = Gnuplot.Gnuplot(persist=1)
    p1_ex = Gnuplot.PlotItems.Data(np.c_[d.tpoints[::points] * d.faketau, d.r_ex[::points, d.l / 2] / d.faketau],
                                   with_='lines')
    p1_in = Gnuplot.PlotItems.Data(np.c_[d.tpoints[::points] * d.faketau, d.r_in[::points, d.l / 2] / d.faketau],
                                   with_='lines')
    s1_ex = Gnuplot.PlotItems.Data(np.c_[d.tpoints[::points] * d.faketau, d.s_ex[::points, d.l / 2] / d.faketau],
                                   with_='lines')
    s1_in = Gnuplot.PlotItems.Data(np.c_[d.tpoints[::points] * d.faketau, d.s_in[::points, d.l / 2] / d.faketau],
                                   with_='lines')
    if args.s != 'nf':
        p2 = Gnuplot.PlotItems.Data(
            np.c_[np.array(fr.tempsfr) * d.faketau, np.array(fr.r)[::points, d.l / 2] / d.faketau],
            with_='lines')
    else:
        p2 = Gnuplot.PlotItems.Data(np.c_[d.tpoints[::points] * d.faketau, d.it[::points, d.l / 2] + d.r0 / d.faketau],
                                    with_='lines')
    # gp.plot(p1_ex, p1_in, p1_exin)
    gp.plot(p1_ex, p1_in, p2)
    # gp2 = Gnuplot.Gnuplot(persist=1)
    # gp2.plot(p1v_ex, p1v_in)
    gp3 = Gnuplot.Gnuplot(persist=1)
    # gp3.plot(p11_ex, p11_in, p11_exin)
    raw_input("Enter to exit ...")

    # np.savetxt("p%d.dat" % args.m, np.c_[d.tpoints[::points] * d.faketau, d.r[::points, d.l / 2] / d.faketau])
