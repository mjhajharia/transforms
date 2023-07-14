import os
from tqdm import tqdm
import sys
sys.path.insert(1, 'utils')

from sample import sample
from utils import create_param_map, list_transforms, list_params
import pickle
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--parameters', type=int, required=True)
parser.add_argument('--transform', type=str, required=True)
parser.add_argument('--inits', type=float, required=False)
args = parser.parse_args()

param_map = create_param_map()
alpha = param_map[args.parameters]
print(alpha)
N = len(alpha)
params={'alpha':alpha, 'N':N}

sample(transform_category='simplex', transform=args.transform, evaluating_model='DirichletSymmetric', 
parameters=params, auto_eval_all_params=False, n_iter = 1000, n_chains = 4, n_repeat=100,
show_progress = True, resample=True, return_idata=False,
        output_dir= '/mnt/home/mjhajaria/ceph/', inits=args.inits)
