

import os
os.chdir('transforms')

import sys
sys.path.insert(1, 'utils')
from sample import sample

import pickle
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--parameters', type=int, required=True)
args = parser.parse_args()

with open(f"target_densities/param_map_dirichlet_symmetric.json", "rb") as f:
    param_map = pickle.load(f)

alpha = param_map[args.parameters]
N = len(alpha)

import numpy as np
import matplotlib.pyplot as plt
import arviz as az

from dirichlet_symmetric import get_dirichlet_symmetric_rmse
from rhat import calc_rhat_mixed_chains
from ess import get_ess_leapfrog_ratio

transform_category='simplex'
evaluating_model='dirichlet_symmetric'

transforms = ['softmax', 'softmax-augmented','stickbreaking', 'stan']

get_dirichlet_symmetric_rmse(transforms, transform_category, 
                             parameters=[{'alpha':alpha, 'N':N}], fig_name='rmse_1.png')

