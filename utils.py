import pandas as pd
import numpy as np
import os
import json
import sys
import pickle
from cmdstanpy import CmdStanModel
import argparse
import arviz as az
from pathlib import Path
from tqdm import tqdm



def sample(transform_category, transform, evaluating_model, parameters, output_file=None, 
                auto_eval_all_params=False, n_iter = 1000, n_chains = 4, n_repeat = 1, show_progress = True):
    '''
    Sample from the given dictionary containing the models, transform_categories, and parameters.
    It saves the results in a json file which is named with n_repeat, n_chains and n_iter.

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
        e.g. parameters = [{'alpha': [0.1]*10, 'K': 10}, {'alpha': [0.1]*20, 'K': 20}]
    
    output_file: str
        Custom filename.

    auto_eval_all_params: bool
        If True, all parameterizations are evaluated.

    n_repeat: int
        Number of times to repeat the sampling.

    n_iter : int
        Number of samples to be drawn.

    n_chains : int
        Number of chains to be drawn.

    show_progress: bool (default = True)
        Whether to show progress bar.

    Note
    ----
    The directory structure for storing stan files is: transform_category/transform/model_name.stan
    The directory structure for storing the results is: sampling_results/transform_category/transform/model_name/{n_iter}_iters_{n_chains}_chains_{n_repeat}_repeats.json
    '''
    if auto_eval_all_params:
        with open(f'{transform_category}/{evaluating_model}.json', 'rb') as f:
            parameters = pickle.load(f)
    
    with open(f'{transform_category}/parameters.json', 'rb') as f:
        param_map = pickle.load(f)

    path=f'sampling_results/{transform_category}/{transform}/{evaluating_model}'
    os.makedirs(path, exist_ok=True)

    for params in tqdm(parameters):
        model = CmdStanModel(stan_file = f'{transform_category}/{transform}/{evaluating_model}.stan', cpp_options = {'STAN_THREADS':'true'})
        idata = az.from_cmdstanpy(model.sample(data = params, show_progress = show_progress, iter_sampling = n_iter, chains = n_chains))

        for i in range(n_repeat-1):
            fit = model.sample(data = params, show_progress = show_progress, iter_sampling = n_iter, chains = n_chains)
            idata = az.concat(idata, az.from_cmdstanpy(fit), dim="chain")

        filename = output_file if output_file else f'{param_map[tuple(list(params.values())[0])]}_{n_repeat}.json'
        Path(f'{path}{filename}').unlink(missing_ok=True)
        idata.to_json(f'{path}/{filename}')

    pass

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))

def cumulative_mean(x):
    return np.divide(np.cumsum(x), np.arange(1, len(x) + 1))

def rmse_leapfrog(idata, true_var, var_name, var_dim):
    cumulative_leapfrog_steps = np.cumsum(np.mean(idata.sample_stats.n_steps, axis=0))
    pred_var = cumulative_mean(np.mean(idata.posterior[str(var_name)].sel(x_dim_0=var_dim), axis=0))
    rmse_array=[]
    for i in range(1, len(pred_var)+1):
        rmse_array.append(rmse(true_var[var_dim], pred_var[:i]))
    return cumulative_leapfrog_steps, rmse_array

def retrieve(transform_category, transform, evaluating_model, parameters, n_repeat):
    '''
    Retrieve the sampling results from the json file.
    
    Parameters
    ----------
    transform_category: str
        The transform category.
    
    transform: str
        The transform name.
    
    evaluating_model: str
        The model.
    
    parameters: list of dict
        Dictionary containing the parameters used.
    
    n_repeat: int  
        Number of times sampling was repeated.
    
    Returns
    -------
    sampling_results: idata object
    '''
        
    with open(f'{transform_category}/parameters.json', 'rb') as f:
        param_map = pickle.load(f)

    filename = f'sampling_results/{transform_category}/{transform}/{evaluating_model}/{param_map[tuple(list(parameters.values())[0])]}_{n_repeat}.json'
    return az.from_json(filename)

