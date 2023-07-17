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
sys.path.insert(1, 'utils/')
from sample import sample

output_dir='/mnt/home/mjhajaria/ceph/sampling_results/simplex'

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--transform', type=str, required=True)
parser.add_argument('--datakey', type=str, required=True)
args = parser.parse_args()

with open('data/dirichletsymmetric.json') as f:
    datajson = json.load(f)


n_repeat=100

stan_filename=f'stan_models/simplex/{args.transform}_DirichletSymmetric.stan'

with open(stan_filename, 'w') as f:
        f.write(f'#include ../../target_densities/DirichletSymmetric.stan{os.linesep}#include ../../transforms/simplex/{args.transform}.stan{os.linesep}')
        f.close()
        
output_file_name=f'{output_dir}/{args.transform}/DirichletSymmetric/draws_{args.datakey}_{n_repeat}.nc'
alpha=datajson[args.datakey]
data={'alpha': alpha, 'N': len(alpha)}

output_file_name_time=f'{output_dir}/{args.transform}/DirichletSymmetric/time_{args.datakey}_{n_repeat}.txt'

sample(
stan_filename,
data,
output_file_name,
n_repeat,
output_file_name_time,
n_iter=1000,
n_chains=4,
show_progress=True,
return_idata=False,
inits=None
)
