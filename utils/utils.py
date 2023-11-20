import pandas as pd
import numpy as np
import os
import time
import json
import sys
import pickle
from cmdstanpy import CmdStanModel
import argparse
import arviz as az
from pathlib import Path
from tqdm import tqdm
from pathlib import Path
from scipy.stats import norm, entropy


# import logging
# logger = logging.getLogger('cmdstanpy')
# logger.addHandler(logging.NullHandler())
def list_transforms():
    return ['Stickbreaking', 'ALR', 'NormalizedExponential'
    'AugmentedILR', 'HypersphericalAngular', 'HypersphericalLogit', 'StanStickbreaking',
    'HypersphericalProbit', 'ProbitProduct', 'AugmentedSoftmax']

def transforms_labels():
    return {'Stickbreaking': 'Stick-breaking',
                    'ALR': 'Additive Log Ratio',
                    'AugmentedSoftmax': 'Augmented-Softmax',
                    'StanStickbreaking': 'Stick-breaking (in C++)',
                    'AugmentedILR'  : 'Augmented-Isometric Log Ratio',
                    'HypersphericalProbit': 'Hyperspherical - Probit',
                    'HypersphericalAngular': 'Hyperspherical-Angular',
                    'HypersphericalLogit': 'Hyperspherical-Logit',
                    'ProbitProduct': 'Probit Product',
                    }


def dirichletsymmetricparams():
    return [{'alpha':[0.1]*10, 'N':10}, {'alpha':[0.1]*100, 'N':100}, {'alpha': [0.1]*1000, 'N': 1000},
                    {'alpha':[1]*10, 'N':10}, {'alpha':[1]*100, 'N':100},  {'alpha': [1]*1000, 'N': 1000},
                    {'alpha':[10]*10, 'N':10}, {'alpha':[10]*100, 'N':100},  {'alpha': [10]*1000, 'N': 1000}]

def get_kl_divergence(draws, n_iter):
    l = draws.draws_pd()[['x[1]']].values[:n_iter]
    rv = norm()
    r = norm.rvs(size=n_iter)
    return entropy(l, r)[0]

def getleapfrog(idata):
    return np.cumsum(np.mean(idata.sample_stats.n_steps, axis=0))
