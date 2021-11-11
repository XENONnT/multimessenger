#!/usr/bin/python

import warnings
warnings.filterwarnings('ignore')


import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import matplotlib
from matplotlib import cm
from matplotlib.colors import LogNorm
import seaborn as sns

import scipy.integrate as integrate
import scipy as sp
import scipy.interpolate as itp
from scipy.stats import gaussian_kde, norm, skewnorm
from scipy.optimize import curve_fit
from scipy.integrate import quad, trapz
from scipy.special import spherical_jn
from scipy.integrate import quad, trapz

import astropy.units as u

def isnotebook():
    """ Tell if the script is running on a notebook

    """
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True  # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter

# from .sn_utils import isnotebook
if isnotebook(): from tqdm.notebook import tqdm
else: from tqdm import tqdm
import pandas as pd
import _pickle as pickle # new/ faster
import numpy as np
import multihist as mh
import click
import functools

import strax, straxen, wfsim, cutax
import nestpy
import datetime, os
import configparser

config = configparser.ConfigParser()
config.read('/dali/lgrandi/melih/mma/data/basic_conf.conf')
# config.read('../data/basic_conf.conf')

path_img       = config['paths']['imgs']
path_data      = config['paths']['data']
paths = {'img':path_img, 'data':path_data}

plt.style.use('ggplot')

plt.rcParams['xtick.labelsize']=12
plt.rcParams['ytick.labelsize']=12
plt.rcParams['xtick.direction']='out'
plt.rcParams['ytick.direction']='out'

plt.rcParams['xtick.major.size']=10
plt.rcParams['ytick.major.size']=10
plt.rcParams['xtick.major.pad']=5
plt.rcParams['ytick.major.pad']=5

plt.rcParams['xtick.minor.size']=5
plt.rcParams['ytick.minor.size']=5
plt.rcParams['xtick.minor.pad']=5
plt.rcParams['ytick.minor.pad']=5
plt.rcParams['legend.fontsize']=18
plt.rcParams['font.size']=16

font_small = {'family': 'serif',
        'color':  'darkred',
        'weight': 'normal',
        'size': 16,
        }

font_medium = {'family': 'serif',
        'color':  'darkred',
        'weight': 'normal',
        'size': 20,
        }

font_large = {'family': 'serif',
        'color':  'darkred',
        'weight': 'normal',
        'size': 24,
        }
params = {'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large',
         }
# Updates plots to apply the above formatting to all plots in doc
pylab.rcParams.update(params)