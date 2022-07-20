import os
from tqdm import tqdm
os.chdir('transforms')

import sys
sys.path.insert(1, 'utils')

from sample import sample
import pickle
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--parameters', type=int, required=True)
parser.add_argument('--transform', type=str, required=True)
args = parser.parse_args()

with open(f"target_densities/param_map_dirichlet_symmetric.json", "rb") as f:
    param_map = pickle.load(f)

alpha = param_map[args.parameters]
N = len(alpha)

sample(transform_category='simplex', transform=args.transform, evaluating_model='dirichlet_symmetric', 
parameters={'alpha':alpha, 'N':N}, 
            auto_eval_all_params=False, n_iter = 1000, n_chains = 4, n_repeat = 100, 
                            show_progress = True, resample=True, return_idata=False)


