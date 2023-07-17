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

transforms = ['StanStickbreaking', 'Stickbreaking', 'ALR','AugmentedSoftmax',
    'AugmentedILR', 'HypersphericalAngular', 'HypersphericalLogit',
    'HypersphericalProbit', 'ProbitProduct']

n_repeat=100
for transform in transforms:
    for datakey in ['3', '6', '9']:
    
        output_file_name=f'{output_dir}/{transform}/DirichletSymmetric/draws_{datakey}_{n_repeat}.nc'
        alpha=datajson[datakey]
        data={'alpha': alpha, 'N': len(alpha)}

        output_file_name_time=f'{output_dir}/{transform}/DirichletSymmetric/time_{datakey}_{n_repeat}.txt'

        try:
            idata = az.from_netcdf(output_file_name)

            true_var = np.asarray(data['alpha'])/sum(data['alpha'])

            def cumulative_mean(x):
                return np.divide(np.cumsum(x), np.arange(1, len(x) + 1))

            rmse={}
            for i in [0,10,20,30,40,50,60,70,80,99]:
                pred_var = cumulative_mean(np.mean(idata.posterior['x'].sel(x_dim_0=i), axis=0))

                rmse_array = []
                for j in tqdm(range(1, len(pred_var) + 1)):
                    rmse_array.append(np.sqrt(np.mean((true_var[i]-pred_var[:j].values) ** 2)))
                rmse['x_'+str(i)] = rmse_array
                
            with open(f'{output_dir}/{transform}/DirichletSymmetric/rmse_{datakey}_{n_repeat}.pickle', 'wb') as handle:
                pickle.dump(rmse, handle, protocol=pickle.HIGHEST_PROTOCOL)
        except OSError:
            print(transform, datakey, 'not found')