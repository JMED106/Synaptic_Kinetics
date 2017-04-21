#!/usr/bin/python2.7
import getopt
import yaml
import sys
import logging.config
from colorlog import ColoredFormatter
import numpy as np

from tools import DictToObj

__author__ = 'Jose M. Esnaola Acebes'

""" Load this file to easily handle data saved in main. It loads the numpy object saved
    in ./results and converts it to a Dict. The latter is converted to an object to
    be able to use the dot notation, instead of the brackets typical of dictionaries.
    To use it in python: run load_data.py -f <name_of_file>
"""

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
handler.setFormatter(formatter)
logger = logging.getLogger('simulation')

pi = np.pi


def __init__(argv):
    reduced = False
    filein = None
    try:
        opts, args = getopt.getopt(argv, "hf:r:", ["file=", "reduced="])
    except getopt.GetoptError:
        print 'load_data.py [-f <file>]'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print 'load_data.py [-f <file>]'
            sys.exit()
        elif opt in ("-f", "--file"):
            filein = arg
        elif opt in ("-r", "--reduced"):
            reduced = arg
        else:
            print 'load_data.py [-f <file>]'
            sys.exit()
    return filein, reduced

data = None


def loadata(ofile, dat):
    del dat
    d = np.load(ofile)
    return DictToObj(d.item())


fin, reduced = __init__(sys.argv[1:])
fin2 = np.array(map(str, fin))
data = loadata(fin, data)
phi = np.linspace(-pi, pi, data.parameters.l)
phip = np.linspace(-pi, pi, data.parameters.l + 1)

index = int(np.argwhere(fin2 == '_'))
results = 'results/'
system = "".join(fin2[len('results/'):index])
logger.info('System: %s' % system)
mode = int(fin2[index + 2])
logger.info('Mode: %s' % mode)
x = t = None


if system == 'nf':
    x = (data.nf.fr.ring.ex + data.nf.fr.ring.inh) / 2.0 / data.parameters.tau
    t = data.nf.t * data.parameters.tau

if system in ('noise', 'gauss'):
    x = data.qif.fr.ring / data.parameters.tau
    t = np.array(data.qif.t) * data.parameters.tau

x_mean = x.mean(axis=1)
x_filter = x - np.dot(x_mean.reshape((len(x_mean), 1)), np.ones((1, data.parameters.l)))
if reduced and system == 'nf':
    system = 'nfreduced'
if system == 'nfreduced':
    np.save("./time_series/mode%d_%s_t.npy" % (mode, system), t[::100])
    np.save("./time_series/mode%d_%s_r.npy" % (mode, system), x[::100])
    np.save("./time_series/mode%d_%s_r_filtered.npy" % (mode, system), x_filter[::100])
else:
    np.save("./time_series/mode%d_%s_t.npy" % (mode, system), t)
    np.save("./time_series/mode%d_%s_r.npy" % (mode, system), x)
    np.save("./time_series/mode%d_%s_r_filtered.npy" % (mode, system), x_filter)
files = str("./time_series/mode%d_%s" % (mode, system))
logger.info('Successfully saved %s* files' % files)
