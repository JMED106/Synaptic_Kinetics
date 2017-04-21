import getopt
import sys

import matplotlib.pyplot as plt
import numpy as np

from tools import DictToObj, ColorPlot, FrequencySpectrum

__author__ = 'Jose M. Esnaola Acebes'

""" Load this file to easily handle data saved in main. It loads the numpy object saved
    in ./results and converts it to a Dict. The latter is converted to an object to
    be able to use the dot notation, instead of the brackets typical of dictionaries.
    To use it in python: run load_data.py -f <name_of_file>
"""

pi = np.pi


def __init__(argv):
    try:
        opts, args = getopt.getopt(argv, "hf:", ["file="])
    except getopt.GetoptError:
        print 'load_data.py [-f <file>]'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print 'load_data.py [-f <file>]'
            sys.exit()
        elif opt in ("-f", "--file"):
            filein = arg
            return filein
        else:
            print 'load_data.py [-f <file>]'
            sys.exit()

data = None

def loadata(file, data):
    del data
    data = None
    d = np.load(file)
    return DictToObj(d.item())

def cplot(tdata, xdata, num=None, fig=None, tfinal=12, **kwargs):
    if num is None:
        f = []
        ax = []
        p = []
        c = []
        if fig is not None:
            flabel = fig
            if plt.fignum_exists(flabel):
                fg = plt.figure(flabel)
                fg.clf()
        else:
            flabel = 0
        for i, (t, x) in enumerate(zip(tdata, xdata)):
            while plt.fignum_exists(flabel) and fig is None:
                flabel +=1
            f.append(plt.figure(flabel))
            ax.append(f[i].add_subplot(111))
            p.append(ax[i].pcolormesh(t, phip, x.T, cmap=plt.get_cmap('gray'), **kwargs))
            if len(c) == i:
                c.append(plt.colorbar(p[i]))
            else:
                c[i] = plt.colorbar(p[i])
            ax[i].set_ylim([-pi, pi])
            if tfinal is not None:
                ax[i].set_xlim([0.01, 12*data.parameters.tau])
    else:
        f = plt.figure(num)
        ax = f.add_subplot(111)
        p = ax.pcolormesh(tdata[num], phip, xdata[num].T, cmap=plt.get_cmap('gray'))
        c = plt.colorbar(p)
        ax.set_ylim([-pi, pi])
        ax.set_xlim([0.01, 12 * data.parameters.tau])
    return  f, ax, p, c

fin = __init__(sys.argv[1:])

data= loadata(fin, data)
phi = np.linspace(-pi, pi, data.parameters.l)
phip = np.linspace(-pi, pi, data.parameters.l + 1)

x = []
t = []
x_mean = []
x_filter = []
C = []

system = data.parameters.args.s
if system == 'both':
    system = ['nf', 'qif']
else:
    system = [system]

for sys in system:
    if sys == 'nf':
        xdata = (data.nf.fr.ring.ex + data.nf.fr.ring.inh) / 2.0 / data.parameters.tau
        t.append(data.nf.t*data.parameters.tau)
    if sys == 'qif':
        xdata = data.qif.fr.ring/data.parameters.tau
        t.append(np.array(data.qif.t)*data.parameters.tau)
    xdata_mean = xdata.mean(axis=1)
    x_filter.append(xdata - np.dot(xdata_mean.reshape((len(xdata_mean), 1)), np.ones((1, data.parameters.l))))
    x.append(xdata)
    x_mean.append(xdata_mean)


    del xdata, xdata_mean


# We store the time series in a different file (for easier management)
# Comment lines to disable :b
# ts_r = np.array(data.qif.fr.ring) / data.parameters.tau
# ts_t = np.array(data.qif.t) * data.parameters.tau
# np.save("ts_r.npy", ts_r)
# np.save("ts_t.npy", ts_t)
