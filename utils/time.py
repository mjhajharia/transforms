
import os
os.chdir('transforms')
import sys
sys.path.insert(1, 'utils')

import numpy as np
import matplotlib.pyplot as plt
import arviz as az
import time
import pickle 

from sample import sample
from dirichlet_symmetric import get_dirichlet_symmetric_rmse
from rhat import calc_rhat_mixed_chains
from ess import get_ess_leapfrog_ratio

def get_param_map(param):
    with open(f"target_densities/param_map_dirichlet_symmetric.pkl", "rb") as f:
        param_map = pickle.load(f)
    alpha = param_map[param]
    N = len(alpha)
    return alpha[0], N

def timer(transform, param):
    with open(f"target_densities/param_map_dirichlet_symmetric.pkl", "rb") as f:
        param_map = pickle.load(f)
    alpha = param_map[param]
    N = len(alpha)
    start_time = time.time()
    
    sample(transform_category='simplex', transform=transform, evaluating_model='dirichlet_symmetric', 
    parameters={'alpha':alpha, 'N':N}, 
                auto_eval_all_params=False, n_iter = 1000, n_chains = 4, n_repeat = 10, 
                                show_progress = True, resample=True, return_idata=False)

    print(time.time() - start_time)
    return time.time() - start_time




transforms = ['stickbreaking']
timedict={}

for param in [1,2,3,4,5,6,7,8,9]:
    for transform in transforms:
        timetaken = timer(transform,param)
        print(str(transform), timetaken)
        timedict[f'{transform}_{param}'] = timetaken




import json
    
with open("time_sb.json", "w") as f:
    json.dump(timedict, f)


