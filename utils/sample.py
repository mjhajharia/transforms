import os
import time
import csv
import pickle
from cmdstanpy import CmdStanModel
import argparse
from pathlib import Path
from tqdm import tqdm
    

def sample(
    transform,
    target_keyword,
    n_repeat=25,
    n_iter=1000,
    n_chains=4,
    show_progress=True,
    inits=None
):
    stan_filename=f'tmp/{target_keyword}/{transform}.stan'
    path = Path(stan_filename)

    if not path.is_dir():
        path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(stan_filename, 'w') as f:
            f.write(f'#include targetdensities.stan{os.linesep}#include transforms/simplex/{transform}.stan{os.linesep}')
            f.close()

    with open(f'data/{target_keyword}.pickle', 'rb') as file:
        data = pickle.load(file)

    outputpath = Path(f'/mnt/home/mjhajaria/ceph/stan_output/simplex/{target_keyword}')
    if not outputpath.exists():
        outputpath.mkdir(parents=True, exist_ok=True)

    model = CmdStanModel(
        stan_file=stan_filename, 
        cpp_options={"STAN_THREADS": "true"},
        stanc_options={"include-paths":'/mnt/home/mjhajaria/transforms/'})
    for i in tqdm(range(0,n_repeat)):
        start_time = time.time()
        model.sample(
            data=data,
            show_progress=show_progress,
            iter_sampling=n_iter,
            chains=n_chains,
            inits=inits,
            seed=i,
            show_console=True,
            output_dir=f'/mnt/home/mjhajaria/ceph/stan_output/simplex/{target_keyword}'
        )
        
        with open('outputs/time.csv', 'a', newline='') as csvfile:
            fieldnames = ['target_keyword', 'transform', 'seed', 'time']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({
                'target_keyword': target_keyword,
                'transform': transform,
                'seed': i,
                'time': str(time.time() - start_time)
            })
