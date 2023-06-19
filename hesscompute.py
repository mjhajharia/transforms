import bridgestan as bs
from pathlib import Path
import os
from utils import *
import json
import pickle
import arviz as az
import numpy as np
from tqdm import tqdm

n_repeat=100
evaluating_model='DirichletSymmetric'
data_file = lambda *, pkey, n_repeat, transform, evaluating_model: f"/mnt/home/mjhajaria/ceph/sampling_results/simplex/{transform}/{evaluating_model}/samples_{pkey}_{n_repeat}.nc"
stan_filename = lambda *, transform, evaluating_model: f"stan_models/{transform}_{evaluating_model}.stan"
output_filename = lambda *, transform, evaluating_model, x, pkey: f'output/{transform}/{evaluating_model}/{x}_{pkey}.npy'

with open(f"target_densities/param_map_{evaluating_model}.pkl", "rb") as f:
    param_map = pickle.load(f)

for i in range(1,10):
    with open(f'utils/dirichletsymmetricdata/data_{i}.json', 'w') as fp:
        json.dump({'alpha':param_map[i], 'N': len(param_map[i])}, fp)

pkeys=[1,2,3,4,5,6,7,8,9]
transforms=list_transforms()
for pkey in tqdm(pkeys):
    for transform in tqdm(transforms):            
        stan_filename=stan_filename(transform=transform, evaluating_model=evaluating_model)
        with open(stan_filename, 'w') as f:
            f.write(f'#include target_densities/DirichletSymmetric.stan{os.linesep}#include transforms/{transform}.stan{os.linesep}')
            f.close()
            
        x = data_file(pkey=pkey, n_repeat=100, 
                transform=transform, evaluating_model='DirichletSymmetric')
        
            
        output_filename=output_filename
        data= f'utils/dirichletsymmetricdata/data_{pkey}.json'
        pkey=pkey
        bsmodel = bs.StanModel.from_stan_file(stan_filename, data, stanc_args=[f"--include-paths='/mnt/home/mjhajaria/simplex-transforms/'"])
        n=bsmodel.param_unc_num()
        print(pkey, bsmodel.param_num(include_tp=True), n)
        hessian=np.empty((400000,n,n))
        grad=np.empty((400000,n))

        try:
            for i in range(400):
                data = az.from_netcdf(x).posterior.sel(chain=[i])
                idata = np.concatenate((data.y.values[0], data.x.values[0]), axis=1)
            
                for idx, row in enumerate(idata):
                    theta = bsmodel.param_unconstrain(row)
                    bsmodel.log_density_hessian(theta, out_grad=grad[idx], out_hess=hessian[idx])
            np.save(output_filename(transform=transform, evaluating_model=evaluating_model, pkey=pkey, x="hessian"), hessian)
            np.save(output_filename(transform=transform, evaluating_model=evaluating_model, 
                                    pkey=pkey, x="grad"), grad)
        except OSError:
            pass