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
import json
import sys
sys.path.insert(1, 'utils')
from sample import sample

output_dir='/mnt/home/mjhajaria/ceph/sampling_results/simplex'

with open('data/dirichletsymmetric.json') as f:
    datajson = json.load(f)

transforms = ['AugmentedSoftmax']


def cumulative_mean(x, axis=0):
    shape = [1] * x.ndim
    shape[axis] = x.shape[axis]
    return np.divide(
        np.cumsum(x, axis=axis), 
        np.reshape(np.arange(1, x.shape[axis] + 1), shape)
    )

n_repeat=100
for transform in transforms:
    for datakey in ['2','5', '8']:
        print(transform, datakey)
        output_file_name=f'{output_dir}/{transform}/DirichletSymmetric/draws_{datakey}_{n_repeat}.nc'
        alpha=datajson[datakey]
        data={'alpha': alpha, 'N': len(alpha)}
        try:
            idata = az.from_netcdf(output_file_name)

            true_alpha = np.asarray(data['alpha'])/sum(data['alpha'])

            rmse={}
            for i in [0, 10, 20, 30, 40, 55, 70, 80, 90, 99]:
                true_var=true_alpha[i]
                pred_var = cumulative_mean(idata.posterior['x'].sel(x_dim_0=i), axis=1)
                rmse_matrix = np.sqrt(cumulative_mean((true_var-pred_var) ** 2))
                rmse['x_'+str(i)] = np.mean(rmse_matrix, axis=0)
                
            with open(f'{output_dir}/{transform}/DirichletSymmetric/rmse_{datakey}_{n_repeat}.pickle', 'wb') as handle:
                pickle.dump(rmse, handle, protocol=pickle.HIGHEST_PROTOCOL)
        except OSError:
            print(transform, datakey, 'not found')