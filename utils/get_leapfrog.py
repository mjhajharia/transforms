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
                print("Skipping")
        else:
            idata = az.from_netcdf(filename=f'/mnt/home/mjhajaria/ceph/sampling_results/simplex/{transform}/DirichletSymmetric/{param}_100.nc')

            leapfrog_steps=np.average(idata.sample_stats['n_steps'].sum(axis=1).values.reshape(-1, 4), axis=1)                
            np.savetxt(f'leapfrog_{transform}_{param}.csv',leapfrog_steps,delimiter =", ")
