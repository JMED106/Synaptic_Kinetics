import datetime
import os

import logging
import numba
import numpy as np
from scipy.fftpack import fft
from scipy.signal import argrelextrema, welch, butter, lfilter, chirp, hilbert
import matplotlib.pyplot as plt

from nflib import Data, Connectivity

logging.getLogger('tools').addHandler(logging.NullHandler())

__author__ = 'Jose M. Esnaola Acebes'

""" Library containing different tools to use alongside the main simulation:

    + Perturbation class: creates a perturbation profile and handles perturbation timing.
    + Saving Class: contains a method to create a dictionary with the data and a method to save it.
    + Theoretical computation class: some computations based on theory.
    + A class that transforms a python dictionary into a python object (for easier usage).

"""

# Constants
pi = np.pi
pi2 = np.pi * np.pi


# Function that performs the integration (prepared for numba)
@numba.autojit
def qifint(v_exit_s1, v, exit0, eta_0, s_0, tiempo, number, dn, dt, tau, vpeak, refr_tau, tau_peak):
    """ This function checks (for each neuron) whether the neuron is in the
    refractory period, and computes the integration in case is NOT. If it is,
    then it adds a time step until the refractory period finishes.

    The spike is computed when the neuron in the refractory period, i.e.
    a neuron that has already crossed the threshold, reaches the midpoint
    in the refractory period, t_peak.
    :rtype : object
    """

    d = 1 * v_exit_s1
    # These steps are necessary in order to use Numba
    t = tiempo * 1.0
    for n in xrange(number):
        d[n, 2] = 0
        if t >= exit0[n]:
            d[n, 0] = v[n] + (dt / tau) * (v[n] * v[n] + eta_0[n] + tau * s_0[int(n / dn)])  # Euler integration
            if d[n, 0] >= vpeak:
                d[n, 1] = t + refr_tau - (tau_peak - 1.0 / d[n, 0])
                d[n, 2] = 1
                d[n, 0] = -d[n, 0]
    return d


@numba.autojit
def qifint_noise(v_exit_s1, v, exit0, eta_0, s_0, nois, tiempo, number, dn, dt, tau, vpeak, refr_tau, tau_peak):
    d = 1 * v_exit_s1
    # These steps are necessary in order to use Numba (don't ask why ...)
    t = tiempo * 1.0
    for n in xrange(number):
        d[n, 2] = 0
        if t >= exit0[n]:
            d[n, 0] = v[n] + (dt / tau) * (v[n] * v[n] + eta_0 + tau * s_0[int(n / dn)]) + nois[n]  # Euler integration
            if d[n, 0] >= vpeak:
                d[n, 1] = t + refr_tau - (tau_peak - 1.0 / d[n, 0])
                d[n, 2] = 1
                d[n, 0] = -d[n, 0]
    return d


def noise(length=100, disttype='g'):
    if disttype == 'g':
        return np.random.randn(length)


def find_nearest(array, value, ret='id'):
    """ find the nearest value or/and id of an "array" with respect to "value"
    """
    idx = (np.abs(array - value)).argmin()
    if ret == 'id':
        return idx
    elif ret == 'value':
        return array[idx]
    elif ret == 'both':
        return array[idx], idx
    else:
        print "Error in find_nearest."
        return -1


class Perturbation:
    """ Tool to handle perturbations: time, duration, shape (attack, decay, sustain, release (ADSR), etc. """
    def __init__(self, data=None, t0=2.5, dt=0.5, ptype='pulse', stype='fourier', modes=None, amplitude=1.0, attack='exponential',
                 release='instantaneous', cntmodes=None, duration=None):
        self.logger = logging.getLogger('tools.Perturbation')
        if data is None:
            self.d = Data()
        else:
            self.d = data

        if modes is None:  # Default mode perturbation is first mode
            modes = [1]
        self.logger.debug("Modes of perturbation: %s" % str(modes))

        # Input at t0
        self.input = np.ones(self.d.l) * 0.0
        # Input time series
        self.it = np.zeros((self.d.nsteps, self.d.l))
        # Input ON/OFF
        self.pbool = False

        # Time parameters
        self.t0 = t0
        self.t0step = int(t0 / self.d.dt)
        self.dt = dt
        self.tf = t0 + dt
        self.t = 0
        self.duration = duration
        self.period = 0.0
        # Rise variables (attack) and parameters
        self.attack = attack
        self.taur = 0.2
        self.trmod = 0.01
        self.trmod0 = 1.0 * self.trmod
        # Decay (release) and parameters
        self.release = release
        self.taud = 0.2
        self.tdmod = 1.0
        self.mintd = 0.0
        # Oscillatory forcing:
        self.chirp = chirp(data.tpoints*data.faketau, 0.0, data.tpoints[-1]*data.faketau, 60.0, phi=-90)
        self.freq = []

        # Amplitude parameters
        self.ptype = ptype
        self.spatialtype = stype
        self.amp = amplitude
        # Spatial modulation (wavelengths)
        self.random = False
        self.phi = np.linspace(-np.pi, np.pi, self.d.l)
        self.smod = self.sptprofile(modes, self.amp, cntmodes=cntmodes)

        # Noisy perturbation: auxiliary matrix
        self.auxMat = np.zeros((self.d.l, self.d.l / 10))
        for i in xrange(self.d.l / 10):
            self.auxMat[i * self.d.l / 10:(i + 1) * self.d.l / 10, i] = 1.0

        # Displacement perturbation: auxiliary matrix
        if 'qif' in self.d.systems:
            self.auxMatD = np.zeros((self.d.Ne, self.d.l))
            for i in xrange(self.d.l):
                self.auxMatD[i * self.d.l:(i + 1) * self.d.l, i] = 1.0

    def sptprofile(self, modes, amp=1E-2, cntmodes=None):
        """ Gives the spatial profile of the perturbation: different wavelength and combinations
            of them can be produced.
        """
        self.logger.debug("Spatial profile of the perturbation: '%s'" % self.spatialtype)
        if self.spatialtype == 'gauss':
            return amp*Connectivity.vonmises(1.0, 8.0, 0.0, 1.0, self.d.l, np.linspace(-pi ,pi, self.d.l))
        elif self.spatialtype == 'fourier':
            sprofile = 0.0
            phi = 0.0
            if np.isscalar(modes):
                self.logger.warning("'Modes' should be an iterable.")
                modes = [modes]
            for m in modes:
                if cntmodes is None:
                    if self.random:
                        phi = np.random.randn(1) * np.pi
                    else:
                        phi = 0.0
                    self.logger.debug("Perturbation of mode %d with phase %f" % (m, phi))
                    sprofile += amp * np.cos(m * self.phi + phi)
                else:
                    sprofile += amp * cntmodes[m]
            return sprofile
        else:
            self.logger.error("Spatial profile '%s' not implemented." % self.spatialtype)
            return 0.0

    def timeevo(self, temps, freq=1.0):
        """ Time evolution of the perturbation """
        # Single pulse
        if self.ptype == 'pulse':
            # Release time, after tf
            if temps >= self.tf:
                if self.release == 'exponential' and self.tdmod > self.mintd:
                    self.tdmod -= (self.d.dt / self.taud) * self.tdmod
                    self.input = self.tdmod * self.smod
                elif self.release == 'instantaneous':
                    self.input = 0.0
                    self.tdmod = 1.0
                    self.trmod = 0.01
                    self.pbool = False
                    self.t0 += self.period
                    self.tf += self.period
            else:  # During the pulse (from t0 until tf)
                if self.attack == 'exponential' and self.trmod < 1.0:
                    self.trmod += (self.d.dt / self.taur) * self.trmod
                    self.tdmod = self.trmod
                    self.input = (self.trmod - self.trmod0) * self.smod
                elif self.attack == 'instantaneous':
                    if temps >= self.t0:
                        self.input = 1.0 * self.smod
        elif self.ptype == 'oscillatory':
            self.freq.append(freq)
            if temps >= self.t0 + self.dt:
                self.trmod = self.amp * np.sin(self.t * 1.0 * freq * 2.0 * np.pi)
                self.t += self.d.dt
            if temps >= self.t0 + self.dt + self.duration:
                self.trmod = 0.0
            self.input = self.trmod * self.smod
        elif self.ptype == 'chirp':
            tstep = int(temps/self.d.dt)
            self.trmod = self.amp * self.chirp[tstep]
            self.input = self.trmod * self.smod


class SaveResults:
    """ Save Firing rate data to be plotted or to be loaded with numpy.
    """
    def __init__(self, data=None, cnt=None, pert=None, path='results', system='nf', parameters=None):
        self.logger = logging.getLogger('tools.SaveResults')
        if data is None:
            self.d = Data()
        else:
            self.d = data
        if cnt is None:
            self.cnt = Connectivity()
        else:
            self.cnt = cnt
        if pert is None:
            self.p = Perturbation(data=data, ptype='none')
        else:
            self.p = pert

        # Path of results (check path or create)
        if os.path.isdir("./%s" % path):
            self.path = "./results"
        else:  # Create the path
            os.path.os.mkdir("./%s" % path)
        # Define file paths depending on the system (nf, qif, both)
        self.fn = SaveResults.FileName(self.d, system)
        self.results = dict(parameters=dict(), connectivity=dict)
        # Parameters are store copying the configuration dictionary and other useful parameters (from the beginning)
        self.results['parameters'] = {'l': self.d.l, 'eta0': self.d.eta0, 'delta': self.d.delta, 'j0': self.d.j0,
                                      'tau': self.d.faketau, 'args': parameters}
        self.results['connectivity'] = {'type': cnt.profile, 'cnt_ex': cnt.cnt_ex, 'cnt_in': cnt.cnt_in,
                                        'cnt': cnt.cnt_ex + cnt.cnt_in,
                                        'eigenmodes': cnt.eigenmodes,
                                        'eigenvectors': cnt.eigenvectors, 'freqs': cnt.freqs}
        self.results['perturbation'] = {}
        for i, pert in enumerate(self.p):
            self.results['perturbation']["%s_%d" % (pert.ptype, i)] = {'t0': pert.t0}
        if cnt.profile == 'mex-hat':
            self.results['connectivity']['je'] = cnt.je
            self.results['connectivity']['ji'] = cnt.ji
            self.results['connectivity']['me'] = cnt.me
            self.results['connectivity']['mi'] = cnt.mi

        if system == 'qif' or system == 'both':
            self.results['qif'] = dict(fr=dict(), v=dict())
            self.results['parameters']['qif'] = {'N': self.d.N, 'Ne': self.d.Ne, 'Ni': self.d.Ni}
        if system == 'nf' or system == 'both':
            self.results['nf'] = dict(fr=dict(), v=dict())

    def create_dict(self):
        tol = self.d.total_time * (1.0 / 100.0)

        for system in self.d.systems:
            self.results[system]['t'] = self.d.t[system]
            self.results[system]['fr'] = dict(ring=self.d.rstored[system])
            self.results[system]['vstored'] = dict(ring=self.d.vstored[system])
            self.results[system]['fr']['distribution'] = self.d.dr[system]

    def save(self):
        """ Saves all relevant data into a numpy object with date as file-name."""
        now = datetime.datetime.now().timetuple()[0:6]
        sday = "-".join(map(str, now[0:3]))
        shour = "_".join(map(str, now[3:]))
        self.logger.info("Saving data into %s/data_%s-%s" % (self.path, sday, shour))
        np.save("%s/data_%s-%s" % (self.path, sday, shour), self.results)

    def time_series(self, ydata, filename, export=False, xdata=None):
        if export is False:
            np.save("%s/%s_y" % (self.path, filename), ydata)
            if xdata is not None:
                np.save("%s/%s_x" % (self.path, filename), xdata)
        else:  # We save it as a csv file
            np.savetxt("%s/%s.dat" % (self.path, filename), np.c_[xdata, ydata])

    def profiles(self):
        pass

    class FileName:
        """ This class just creates strings to be easily used and understood
            (May be is too much...)
        """

        def __init__(self, data, system):
            self.d = data
            if system == 'qif' or system == 'both':
                self.qif = self.Variables(data, 'qif')
            if system == 'nf' or system == 'both':
                self.nf = self.Variables(data, 'nf')

        @staticmethod
        def tpoints(d, system):
            return "%s_time-colorplot_%.2lf-%.2lf-%.2lf-%d" % (system, d.j0, d.eta0, d.delta, d.l)

        class Variables:
            def __init__(self, data, t):
                self.fr = SaveResults.FileName.FiringRate(data, t)
                self.v = SaveResults.FileName.MeanPotential(data, t)
                self.t = SaveResults.FileName.tpoints(data, t)

        class FiringRate:
            def __init__(self, data, system):
                self.d = data
                self.t = system
                self.colorplot()

            def colorplot(self):
                # Color plot (j0, eta0, delta, l)
                self.cp = "%s_fr-colorplot_%.2lf-%.2lf-%.2lf-%d" % (
                    self.t, self.d.j0, self.d.eta0, self.d.delta, self.d.l)

            def singlets(self, pop):
                # Single populations
                self.sp = "%s_fr-singlets-%d_%.2lf-%.2lf-%.2lf-%d" % (
                    self.t, pop, self.d.j0, self.d.eta0, self.d.delta, self.d.l)

            def profile(self, t0):
                # Profile at a given t0
                self.pr = "%s_fr-profile-%.2lf_%.2lf-%.2lf-%.2lf-%d" % (
                    self.t, t0, self.d.j0, self.d.eta0, self.d.delta, self.d.l)

        class MeanPotential:
            def __init__(self, data, system):
                self.d = data
                self.t = system
                self.colorplot()

            def colorplot(self):
                # Color plot (j0, eta0, delta, l)
                self.cp = "vstored-colorplot_%.2lf-%.2lf-%.2lf-%d" % (self.d.j0, self.d.eta0, self.d.delta, self.d.l)

            def singlets(self, pop):
                # Single populations
                self.sp = "vstored-singlets-%d_%.2lf-%.2lf-%.2lf-%d" % (
                    pop, self.d.j0, self.d.eta0, self.d.delta, self.d.l)

            def profile(self, t0):
                # Profile at a given t0
                self.pr = "vstored-profile-%.2lf_%.2lf-%.2lf-%.2lf-%d" % (
                    t0, self.d.j0, self.d.eta0, self.d.delta, self.d.l)


class TheoreticalComputations:
    def __init__(self, data=None, cnt=None, pert=None):
        if data is None:
            self.d = Data()
        else:
            self.d = data
        if cnt is None:
            self.cnt = Connectivity()
        else:
            self.cnt = cnt
        if pert is None:
            self.p = Perturbation()
        else:
            self.p = pert

        # Theoretical distribution of firing rates
        self.thdist = dict()

    def theor_distrb(self, s, points=10E3, rmax=3.0):
        """ Computes theoretical distribution of firing rates
        :param s: Mean field (J*rstored) rescaled (tau = 1)
        :param points: number of points for the plot
        :param rmax: maximum firing rate of the distribution
        :return: rho(rstored) vs rstored (output -> rstored, rho(rstored))
        """
        rpoints = int(points)
        rmax = rmax / self.d.faketau
        r = np.dot(np.linspace(0, rmax, rpoints).reshape(rpoints, 1), np.ones((1, self.d.l)))
        s = np.dot(np.ones((rpoints, 1)), s.reshape(1, self.d.l))
        geta = self.d.delta / ((self.d.eta0 - (np.pi ** 2 * self.d.faketau ** 2 * r ** 2 - s)) ** 2 + self.d.delta ** 2)
        rhor = 2.0 * np.pi * self.d.faketau ** 2 * geta.mean(axis=1) * r.T[0]
        # plt.plot(x.T[0], hr, 'rstored')
        return dict(x=r.T[0], y=rhor)


class FrequencySpectrum:
    """ Class containing methods for analyzing the frequency spectrum of the system.
           + Decaying frequencies after a perturbation.
           + Frequencies of the system under GWN.
    """

    def __init__(self):
        """ Plotting parameters? Saving parameters?"""
        pass

    @staticmethod
    def analyze(tdata, t0, t1, tau, method='normal'):
        """ This function takes all the time series and performs fft on it."""
        # Data
        num_points = len(tdata)
        logging.info('Analyzing frequency spectrum using Fourier Transform.')

        # t = np.linspace(t0, t1, num_points)
        dt = (t1 - t0) / (1.0 * num_points)
        tf = []
        yf = []
        methods = []
        # FFT data
        if method in ('normal', 'all'):
            methods.append('fft')
            logging.debug('Computing Fourier Transform using fft method')
            tf.append(np.linspace(0, 1.0 / (2.0 * dt * tau), num_points / 2))
            yf.append(2.0 / num_points * np.abs(fft(tdata)[0:num_points / 2]))
        if method in ('welch', 'all'):
            methods.append('welch %s' % 'nperseg=1024 * 70')
            logging.debug('Computing Fourier Transform using welch method')
            # 70 va bastante bien
            yf2 = welch(tdata, fs=1.0 / (tau * dt), nperseg=1024 * 70)
            tf.append(yf2[0])
            yf.append(yf2[1])
        else:
            logging.error('Analysis method not implemented.')
            return -1

        # Normalize measure to be plotted together
        maxx = []
        for y, t, m in zip(yf, tf, methods):
            mask = (t > 5.0)
            mean = np.mean(y)
            maxx.append(np.max(y[mask] / mean))
            plt.plot(t, y/mean, label=m)
        plt.xlim([0, 100])
        plt.ylim([0, np.max(np.array(maxx)*1.1)])
        plt.legend()
        plt.show()
        # exit(-1)
        # index_peaks = np.array(argrelextrema(yf, np.greater))
        # max_peak = np.max(yf[index_peaks])
        # index_freqs = index_peaks[(yf[index_peaks] >= max_peak / 10.0)]
        # freqs_rescaled = tf[index_freqs]
        # freqs = tf[index_freqs] / tau
        # return (yf2[0], yf2[1])

    @staticmethod
    def freqbyhand(tdata, t0, t1, tau, v0=None):
        """ Computes the frequency by detecting the maxims of amplitudes
            in a given time series and dividing by time.
        """
        num_points = len(tdata)
        t = np.linspace(t0, t1, num_points)
        index_peaks = np.array(argrelextrema(tdata, np.greater))[0]
        # print index_peaks
        peaks_time = t[index_peaks]
        # print peaks_time*tau
        freq_rescaled = (len(peaks_time) - 1) / (peaks_time[-1] - peaks_time[0])
        freq = freq_rescaled / tau
        # print freq
        return freq

    @staticmethod
    def plotdata(x, data, xf, dataf):
        plt.plot(x[0:-1:100], data[0:-1:100])
        plt.plot(x, np.log(np.abs(data - data.mean())))
        plt.show()
        plt.plot(xf, dataf)
        plt.grid()
        plt.show()

    @staticmethod
    def butter_bandpass(lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a

    def butter_bandpass_filter(self, data, lowcut, highcut, fs, order=5):
        b, a = self.butter_bandpass(lowcut, highcut, fs, order=order)
        y = lfilter(b, a, data)
        return y

    @staticmethod
    def butter_lowpass(cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a

    def butter_lowpass_filter(self, data, cutoff, fs, order=5):
        b, a = self.butter_lowpass(cutoff, fs, order=order)
        y = lfilter(b, a, data)
        return y

    @staticmethod
    def butter_highpass(cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='high', analog=False)
        return b, a

    def butter_highpass_filter(self, data, cutoff, fs, order=5):
        b, a = self.butter_highpass(cutoff, fs, order=order)
        y = lfilter(b, a, data)
        return y


class LinearStability:
    """ Class containing methods for analyzing the linear stability analysis.
           + Forcing..
           + Envelopes ... .
    """

    def __init__(self):
        """ Things ..."""
        pass

    @staticmethod
    def envelope_amplitude(signal, tau=None):
        """ Method to compute the enveloping amplitude of a signal."""
        if tau is None:
            tau = 20E-3
        samples = np.size(signal)
        duration = samples * tau
        fs = samples / duration
        analytic_signal = hilbert(signal)
        amplitude_envelope = np.abs(analytic_signal)
        instantaneous_phase = np.unwrap(np.angle(analytic_signal))
        instantaneous_frequency = np.diff(instantaneous_phase) / (2.0*np.pi) * fs
        return {'envelope': amplitude_envelope, 'phase': instantaneous_phase, 'freq': instantaneous_frequency}

    @staticmethod
    def envelope2extreme(signal, t, tau=None, pop=None):
        if tau is None:
            tau = 20E-3
        # Filter the signal to a baseline
        l = np.size(signal[0])
        if pop is None:
            pop = l/2
        logging.debug("Shape of 'signal': %s" % str(np.shape(signal)))
        xdata_mean = signal.mean(axis=1)
        x_filter = signal - np.dot(xdata_mean.reshape((len(xdata_mean), 1)), np.ones((1, l)))
        logging.debug("Shape of 'x_filter': %s" % str(np.shape(x_filter)))
        signalpop = x_filter[:, pop]
        logging.debug("Shape of 'signalpop': %s" % str(np.shape(signalpop)))

        extreme_max = argrelextrema(signalpop, np.greater)[0]
        extreme_min = argrelextrema(signalpop, np.less)[0]
        logging.debug("Shapes of 'extremes': %s, %s" % (str(np.shape(extreme_max)), str(np.shape(extreme_min))))
        extreme = np.sort(np.ravel(np.concatenate((extreme_max, extreme_min))))
        logging.debug("Shape of 'extreme': %s" % str(np.shape(extreme)))
        time = t[extreme]*tau
        amplitude = np.abs(signalpop[extreme])
        return {'envelope': amplitude, 't': time, 'args': extreme}


class DictToObj(object):
    """Class that transforms a dictionary d into an object. Note that dictionary keys must be
       strings and cannot be just numbers (even if they are strings:
               if a key is of the format '4', -> put some letter at the beginning,
               something of the style 'f4'.
       Class obtained from stackoverflow: user Nadia Alramli.
 `  """

    def __init__(self, d):
        for a, b in d.items():
            if isinstance(b, (list, tuple)):
                setattr(self, a, [DictToObj(x) if isinstance(x, dict) else x for x in b])
            else:
                setattr(self, a, DictToObj(b) if isinstance(b, dict) else b)


class ColorPlot:
    def __init__(self, data=None, tfinal=12.0):
        self.logger = logging.getLogger('tools.ColorPlot')
        if data is None:
            self.d = Data()
        else:
            self.d = data

        self.phi = np.linspace(-pi, pi, self.d.l)
        self.phip = np.linspace(-pi, pi, self.d.l + 1)

        self.xlim = np.array([self.d.t0, self.d.total_time]) * self.d.faketau
        self.xlimr = np.array([self.d.t0, tfinal]) * self.d.faketau

        self.ylim = [-pi, pi]

        numpoints_max = 1000
        self.step = self.d.nsteps / numpoints_max
        self.stepr = (self.xlimr[-1] - self.xlimr[0]) / self.d.dt / numpoints_max
        if self.stepr < 1:
            self.stepr = 1
        self.logger.debug("Step size: %f\t Step size of the reduce plot: %f" % (self.step, self.stepr))

        self.tpointsr = np.arange(self.d.t0, tfinal, self.d.dt) * self.d.faketau
        self.tsfinal = np.size(self.tpointsr)

    def cplot(self, xdata, density=1.0, paper=True):
        xdata = xdata/self.d.faketau
        if paper:
            step = int(self.stepr / density)
            if step < 1:
                self.logger.warning("Actual Step size: %f" % step)
            pcolor = plt.pcolormesh(self.tpointsr[::step], self.phip, xdata[0:self.tsfinal:step].T, cmap=plt.get_cmap('gray'))
            plt.xlim(self.xlimr)
        else:
            step = int(self.step / density)
            if step < 1:
                self.logger.warning("Actual Step size: %f" % step)
            pcolor = plt.pcolormesh(self.d.tpoints[::step], self.phip, xdata[::step], cmap=plt.get_cmap('gray'))
            plt.xlim(self.xlim)
        plt.ylim(self.ylim)
        cbar = plt.colorbar(pcolor)
        plt.show()

    @staticmethod
    def filter(xdata):
        (t, l) = np.shape(xdata)
        x_mean_phi = xdata.mean(axis=1)
        x_filter = xdata - np.dot(x_mean_phi.reshape((t,1)), np.ones((1,l)))
        return x_filter
