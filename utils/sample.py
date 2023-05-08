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

# import logging
# logger = logging.getLogger('cmdstanpy')
# logger.addHandler(logging.NullHandler())


def sample(
    transform_category,
    transform,
    evaluating_model,
    parameters,
    output_dir='/mnt/sdceph/users/mjhajaria/',
    auto_eval_all_params=False,
    n_iter=1000,
    n_chains=4,
    n_repeat=100,
    show_progress=True,
    resample=False,
    return_idata=True,
    inits=None
):
    """
    Sample from the given dictionary containing the models, transform_categories, and parameters.
    It saves the results in a json file which is named with n_chains.

    Parameters
    ----------
    transform_category: str
        The transform category to use.

    transform: str
        The transform name to use.

    evaluating_model: str
        The model to use.

    parameters: list of dict
        List of Dictionaries containing the parameters to use.
        e.g. parameters = [{'alpha': [0.1]*10, 'N': 10}, {'alpha': [0.1]*20, 'N': 20}]

    output_file: str
        Custom filename.

    auto_eval_all_params: bool
        If True, all parameterizations are evaluated.

    n_iter : int
        Number of samples to be drawn.

    n_chains : int
        Number of chains to be drawn.
    
    n_repeat : int 
        Number of runs to be drawn.

    show_progress: bool (default = True)
        Whether to show progress bar.

    retrieve: bool (default = False)
        Whether to retrieve the results from the json file.

    resample: bool (default = False)
        Whether to resample the data.

    return_idata: bool (default = True)
        Whether to return the idata.
    
    

    Note
    ----
    The directory structure for storing stan files is: stan_models/{transform}_{evaluating_model}.stan
    The directory structure for storing the results is: sampling_results/{transform_category}/{transform}/{evaluating_model}/{param_number}_{n_repeat}.nc

    param_number is obtained from the param_map.pkl file. It is a dictionary with keys as the parameterization and values as the param_number.
    """
    with open(f"target_densities/param_map_{evaluating_model}.pkl", "rb") as f:
        param_map = pickle.load(f)

    if resample == False:
        for params in parameters:
            filename = f'{output_dir}sampling_results/{transform_category}/{transform}/{evaluating_model}/{param_map[tuple(list(params.values())[0])]}_{n_repeat}.nc'
            return az.from_netcdf(filename)
    else:

        if auto_eval_all_params:
            with open(
                f"target_densities/{evaluating_model}_parameters.json", "rb"
            ) as f:
                parameters = pickle.load(f)
       
        stan_filename=f'stan_models/{transform}_{evaluating_model}.stan'

        start_time = time.time()
        if type(parameters)==dict:
            parameters = [parameters]

        Path("stan_models").mkdir(parents=True, exist_ok=True)

        with open(stan_filename, 'w') as f:
            f.write(f'#include ../target_densities/{evaluating_model}.stan{os.linesep}#include ../transforms/{transform_category}/{transform}.stan{os.linesep}')
            f.close()

        for params in tqdm(parameters):
            model = CmdStanModel(
                stan_file=stan_filename, cpp_options={"STAN_THREADS": "true"})
                
            idata = az.from_cmdstanpy(
                model.sample(
                    data=params,
                    show_progress=show_progress,
                    iter_sampling=n_iter,
                    chains=n_chains,
                    inits=inits,
                    show_console=True
                )
            )
            for i in tqdm(range(n_repeat - 1)):
                fit = model.sample(
                    data=params,
                    show_progress=show_progress,
                    iter_sampling=n_iter,
                    chains=n_chains,
                    seed=i,
                    inits=inits,
                    show_console=True
                )
        
                idata = az.concat(idata, az.from_cmdstanpy(fit), dim="chain")
            
            complete_output_dir=f'{output_dir}sampling_results/{transform_category}/{transform}/{evaluating_model}/'
            Path(f'{complete_output_dir}').mkdir(parents=True, exist_ok=True)
            idata.to_netcdf(f'{complete_output_dir}{param_map[tuple(list(params.values())[0])]}_{n_repeat}.nc')
            with open(f'{complete_output_dir}time_{param_map[tuple(list(params.values())[0])]}_{n_repeat}.txt', 'w') as f:
	            f.write(str(time.time() - start_time))

            if return_idata==True:
                return idata