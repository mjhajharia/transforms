import os
from tqdm import tqdm
import sys
sys.path.insert(1, 'utils')

from sample import sample
from utils import create_param_map, list_transforms, list_params
import pickle
import argparse

import arviz as az
import pandas as pd
parser = argparse.ArgumentParser()
parser.add_argument('--transform', type=str, required=True)
args = parser.parse_args()

def create_csv(transform_category, transform, evaluating_model,parameters,n_repeat,output_dir):
        idata = az.from_netcdf(f'/mnt/home/mjhajaria/ceph/sampling_results/{transform_category}/{transform}/{evaluating_model}/{parameters}_{n_repeat}.nc')
        az.summary(idata).to_csv(f'{output_dir}/{transform}_{parameters}_{n_repeat}.csv')


for i in tqdm([1,2,4,5,7,8]):
        for j in ['Stickbreaking', 'ALR', 'AugmentedSoftmax', 'StanStickbreaking', 'AugmentedILR', 'Hyperspherical', 'HypersphericalAngular', 'HypersphericalLogit']:
                create_csv(transform_category='simplex', transform=j, evaluating_model='DirichletSymmetric', 
                parameters=i,n_repeat=100,output_dir= '/mnt/home/mjhajaria/transforms/data/simplex_DirichletSymmetric')

for i in tqdm([3,6,9]):
        for j in tqdm(['Stickbreaking', 'ALR', 'AugmentedSoftmax', 'StanStickbreaking', 'AugmentedILR', 'HypersphericalLogit']):
                create_csv(transform_category='simplex', transform=args.transform, evaluating_model='DirichletSymmetric', 
                parameters=i,n_repeat=100,output_dir= '/mnt/home/mjhajaria/transforms/data/simplex_DirichletSymmetric')

        for j in tqdm(['Hyperspherical', 'HypersphericalAngular']):
                create_csv(transform_category='simplex', transform=j, evaluating_model='DirichletSymmetric', 
                parameters=i,n_repeat=1,output_dir= '/mnt/home/mjhajaria/transforms/data/simplex_DirichletSymmetric')