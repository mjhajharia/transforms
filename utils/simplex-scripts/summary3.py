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

transforms = ['ProbitProduct']

n_repeat=100
for transform in transforms:
    for datakey in ['8']:
    
        # stan_filename=f'../stan_models/simplex/{transform}_DirichletSymmetric.stan'

        # with open(stan_filename, 'w') as f:
        #     f.write(f'#include ../../target_densities/DirichletSymmetric.stan{os.linesep}#include ../../transforms/simplex/{transform}.stan{os.linesep}')
        #     f.close()
        output_file_name=f'{output_dir}/{transform}/DirichletSymmetric/draws_{datakey}_{n_repeat}.nc'
        alpha=datajson[datakey]
        data={'alpha': alpha, 'N': len(alpha)}

        output_file_name_time=f'{output_dir}/{transform}/DirichletSymmetric/time_{datakey}_{n_repeat}.txt'

#         sample(
#         stan_filename,
#         data,
#         output_file_name,
#         n_repeat,
#         output_file_name_time,
#         n_iter=1000,
#         n_chains=4,
#         show_progress=True,
#         return_idata=False,
#         inits=None
#         )
        try:
            idata = az.from_netcdf(output_file_name)

            # ess={}
            # for i in idata.posterior.x.x_dim_0.values:
            #     ess_array=[]
            #     for k in tqdm(range(n_repeat)):
            #         ess_array.append(list(az.ess(idata.sel(chain=[k*4, k*4+1, k*4+2, k*4+3], 
            #                                         x_dim_0=i), var_names=['x']).values())[0].item())
            #     ess['x_'+str(i)] = ess_array
            
            # with open(f'{output_dir}/{transform}/DirichletSymmetric/ess_{datakey}_{n_repeat}.pickle', 'wb') as handle:
            #     pickle.dump(ess, handle, protocol=pickle.HIGHEST_PROTOCOL)
                
            # true_var = np.asarray(data['alpha'])/sum(data['alpha'])

            # def cumulative_mean(x):
            #     return np.divide(np.cumsum(x), np.arange(1, len(x) + 1))

            # rmse={}
            # for i in idata.posterior.x.x_dim_0.values:
            #     pred_var = cumulative_mean(np.mean(idata.posterior['x'].sel(x_dim_0=i), axis=0))

            #     rmse_array = []
            #     for j in tqdm(range(1, len(pred_var) + 1)):
            #         rmse_array.append(np.sqrt(np.mean((true_var[i]-pred_var[:j].values) ** 2)))
            #     rmse['x_'+str(i)] = rmse_array
                
            # with open(f'{output_dir}/{transform}/DirichletSymmetric/rmse_{datakey}_{n_repeat}.pickle', 'wb') as handle:
            #     pickle.dump(rmse, handle, protocol=pickle.HIGHEST_PROTOCOL)
                
            with open(f'{output_dir}/{transform}/DirichletSymmetric/leapfrog_{datakey}_{n_repeat}.pickle', 'wb') as handle:
                pickle.dump(idata.sample_stats.n_steps.values, handle, protocol=pickle.HIGHEST_PROTOCOL) 
            
            # if data['N']==1000:
            #     with open(f'{output_dir}/{transform}/DirichletSymmetric/summary_{datakey}_{n_repeat}.pickle', 'wb') as handle:
            #         pickle.dump(az.summary(idata.sel(chain=[0,1,2,3,4,5,6,7,8,9,10,11])), handle, protocol=pickle.HIGHEST_PROTOCOL) 
            # else:
            with open(f'{output_dir}/{transform}/DirichletSymmetric/summary_{datakey}_{n_repeat}.pickle', 'wb') as handle:
                pickle.dump(az.summary(idata), handle, protocol=pickle.HIGHEST_PROTOCOL) 

        except OSError:
            print(transform, datakey, 'not found')