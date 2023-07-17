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
    for datakey in ['1','2','4','5','7','8']:
    
        output_file_name=f'{output_dir}/{transform}/DirichletSymmetric/draws_{datakey}_{n_repeat}.nc'
        alpha=datajson[datakey]
        data={'alpha': alpha, 'N': len(alpha)}

        output_file_name_time=f'{output_dir}/{transform}/DirichletSymmetric/time_{datakey}_{n_repeat}.txt'


        try:
            idata = az.from_netcdf(output_file_name)

            ess=[]
            for k in tqdm(range(n_repeat)):
                ess.append(list(az.ess(idata.sel(chain=[k,k+1,k+2,k+3])).x.values))
            np.save(f'{output_dir}/{transform}/DirichletSymmetric/ess_{datakey}_{n_repeat}.npy', np.asarray(ess))
        
        except OSError:
            print(transform, datakey, 'not found')