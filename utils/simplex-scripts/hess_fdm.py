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
rng = np.random.default_rng(12345)

output_dir='/mnt/home/mjhajaria/ceph/sampling_results/simplex'
with open('data/dirichletsymmetric.json') as f:
    datajson = json.load(f)

n_repeat=100
import bridgestan as bs
bs.set_bridgestan_path('/mnt/home/mjhajaria/.bridgestan/bridgestan-2.0.0')

#diff include paths for bs
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--transform', type=str, required=True)
parser.add_argument('--datakey', type=str, required=True)
args = parser.parse_args()


stan_filename=f'stan_models/sb/{args.transform}_DirichletSymmetric_{args.datakey}.stan'

with open(stan_filename, 'w') as f:
    f.write(f'#include target_densities/DirichletSymmetric.stan{os.linesep}#include transforms/simplex/{args.transform}.stan{os.linesep}')
    f.close()
output_file_name=f'{output_dir}/{args.transform}/DirichletSymmetric/draws_{args.datakey}_{n_repeat}.nc'
alpha=datajson[args.datakey]
data={'alpha': alpha, 'N': len(alpha)}
data = json.dumps(data)

try:
    idata = az.from_netcdf(output_file_name)

    try:
        bsmodel = bs.StanModel.from_stan_file(stan_filename, data, 
        stanc_args=[f"--include-paths='/mnt/home/mjhajaria/transforms/'"],
        make_args=[ "BRIDGESTAN_AD_HESSIAN=false", "STAN_THREADS=true"])
        n=bsmodel.param_unc_num()
        print(bsmodel.param_num(), n)
        if 'BRIDGESTAN_AD_HESSIAN=false' in bsmodel.model_info():
            print("ok")
            cond_array=np.asarray([])
            data= idata.posterior.y.values.reshape(400000,n)
            print(args.datakey, args.transform)
            if args.datakey in ['3', '6', '9']:
                x= list(rng.choice(400000, 4000, replace=False))
                data=data[x]

            if args.datakey in ['2', '5', '8']:
                x= list(rng.choice(400000, 40000, replace=False))
                data=data[x]

            if args.datakey in ['1', '4', '7']:
                data=data
                
            for idx, row in tqdm(enumerate(data)):
                theta = bsmodel.param_unconstrain(row)
                lp, grad, hessian = bsmodel.log_density_hessian(theta)
                try:
                    cond_array= np.append(cond_array, np.linalg.cond(hessian, 2))
                except:
                    cond_array= np.append(cond_array, np.nan)

            np.save(f"{output_dir}/{args.transform}/DirichletSymmetric/fdm_cond_{args.datakey}_{n_repeat}.npy",cond_array)
    except:
        print(f"FDM failed for {args.transform} {args.datakey}")
except OSError:
    print("file not found error")