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
# sys.path.insert(1, '../../')
# from sample import sample

output_dir='/mnt/home/mjhajaria/ceph/sampling_results/simplex'
with open('data/dirichletsymmetric.json') as f:
    datajson = json.load(f)

transforms = [ 'Stickbreaking', 'ALR','HypersphericalAngular', 'HypersphericalLogit',
    'HypersphericalProbit', 'ProbitProduct', 'AugmentedSoftmax']

n_repeat=100
import bridgestan as bs
bs.set_bridgestan_path('/home/meenaljhajharia/.bridgestan/bridgestan-1.0.2')

#diff include paths for bs
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--transform', type=str, required=True)
parser.add_argument('--datakey', type=str, required=True)
args = parser.parse_args()


stan_filename=f'stan_models/{args.transform}_DirichletSymmetric.stan'

with open(stan_filename, 'w') as f:
    f.write(f'#include target_densities/DirichletSymmetric.stan{os.linesep}#include transforms/simplex/{args.transform}.stan{os.linesep}')
    f.close()
output_file_name=f'{output_dir}/{args.transform}/DirichletSymmetric/samples_{args.datakey}_{n_repeat}.nc'
alpha=datajson[args.datakey]
data={'alpha': alpha, 'N': len(alpha)}
data = json.dumps(data)

try:
    idata = az.from_netcdf(output_file_name)

    bsmodel = bs.StanModel.from_stan_file(stan_filename, data, stanc_args=[f"--include-paths='/mnt/home/mjhajaria/transforms/'"])
    n=bsmodel.param_unc_num()
    print(bsmodel.param_num(), n)

    cond_array=np.asarray([])
    data= idata.posterior.y.values.reshape(400000,n)
    for idx, row in tqdm(enumerate(data)):
        theta = bsmodel.param_unconstrain(row)
        lp, grad, hessian = bsmodel.log_density_hessian(theta)
        try:
            cond_array= np.append(cond_array, np.linalg.cond(hessian, 2))
        except:
            cond_array= np.append(cond_array, np.nan)

    np.save(f"{output_dir}/{args.transform}/DirichletSymmetric/cond_{args.datakey}_{n_repeat}.npy",cond_array)

except OSError:
    print("file not found error")