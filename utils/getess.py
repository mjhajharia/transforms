#!/usr/bin/env python
# coding: utf-8


import numpy as np
from tqdm import tqdm
import arviz as az
transforms =  ['Stickbreaking', 'ALR', 'AugmentedSoftmax', 'StanStickbreaking', 
        'AugmentedILR', 'HypersphericalAngular', 'HypersphericalLogit',
         'HypersphericalProbit', 'ProbitProduct']
params=[1,2,3,4,5,6,7,8,9]


for transform in tqdm(transforms):
    for param in tqdm(params):
        if (param==3 and transform=='HypersphericalAngular') or (param==6 and transform=='HypersphericalAngular') or (param==3 and transform=='ProbitProduct'):
                print("nit this")
        else:
            ess=[]
            idata = az.from_netcdf(filename=f'/mnt/home/mjhajaria/ceph/sampling_results/simplex/{transform}/DirichletSymmetric/{param}_100.nc')

            for k in tqdm(range(100)):
                ess.append(list(az.ess(idata.sel(chain=[k*4, k*4+1, k*4+2, k*4+3], x_dim_0=0), var_names=['x']).values())[0].item())
                
            np.savetxt(f'ess_{transform}_{param}.csv',np.asarray(ess),delimiter =", ")
