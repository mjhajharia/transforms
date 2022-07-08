import cmdstanpy
import pandas as pd
import arviz as az
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import os
from cmdstanpy import cmdstan_path, CmdStanModel


simplex_stan = os.path.join('simplex-stan/simplex-stan.stan')
simplex_stickbreaking = os.path.join('simplex-stickbreaking/simplex-stickbreaking.stan')
simplex_softmax = os.path.join('simplex-softmax/simplex-softmax.stan')

def calc_rhat_mixed_chains(path_1, path_2, variables, data, force_compile=False):
    
    """
    This function evaluates rhat for 4 chains from 2 different models, as in 2 chains from each.
    
    Parameters
    ----------
    
    path_1: str
        path for the first model
        
    path_2: str
        path for the second model
    
    variables: list
        Variables to evaluate rhat for
    
    data: dict
        Dictionary of data
    
    force_compile: Bool
        Whether to recompile or not
    """
    variables = ' '.join(variables)
    #Build stan model
    file_1 = os.path.join(path_1)
    file_2 = os.path.join(path_2)
    model_1 = CmdStanModel(stan_file=file_1, cpp_options={'STAN_THREADS':'true'})
    model_2 = CmdStanModel(stan_file=file_2, cpp_options={'STAN_THREADS':'true'})
    
    #Recompile
    if force_compile is True:
        model_1.compile(force=True)
        model_2.compile(force=True)
    
    #Fit stan model
    fit_1 = model_1.sample(data=dict(K=10))
    fit_2 = model_2.sample(data=dict(K=10))
    
    #Convert to idata for arviz
    idata_1 = az.from_cmdstanpy(fit_1)
    idata_2 = az.from_cmdstanpy(fit_2)
    
    #Stack the samples according to chains
    stacked_1 = az.extract_dataset(idata_1)
    stacked_2 = az.extract_dataset(idata_2)
    
    #Concatenate chains and evaluate rhat
    chains = np.concatenate((stacked_1.sel(chain=1)[str(variables)], stacked_1.sel(chain=2)[str(variables)], stacked_2.sel(chain=1)[str(variables)], stacked_2.sel(chain=2)[str(variables)]), axis=1)
    rhat = az.rhat(chains, var_names=variables,method="rank")
    return rhat