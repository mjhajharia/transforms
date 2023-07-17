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


def sample(
    stan_filename,
    data,
    output_file_name,
    n_repeat=100,
    output_file_name_time=None,
    n_iter=1000,
    n_chains=4,
    show_progress=True,
    return_idata=False,
    inits=None
):
    """
    Sample from the given dictionary containing the models, transform_categories, and parameters.
    It saves the results in a json file which is named with n_chains.

    Parameters
    ----------
    stan_filename: str
        The stan file to use.

    data: dict
        Dictionary of data for CmdStanPy Model. 

    output_file_name: str
        Custom filename to save the results

    n_iter : int
        Number of samples to be drawn.

    n_chains : int
        Number of chains to be drawn.
    
    n_repeat : int 
        Number of runs to be drawn.

    output_file_name_time: str (default = None)
        Custom filename to save the time.

    show_progress: bool (default = True)
        Whether to show progress bar.

    retrieve: bool (default = False)
        Whether to retrieve the results from the json file.

    return_idata: bool (default = True)
        Whether to return the idata.
    """
    start_time = time.time()

    model = CmdStanModel(
        stan_file=stan_filename, cpp_options={"STAN_THREADS": "true"})
    
    idata = az.from_cmdstanpy(
        model.sample(
            data=data,
            show_progress=show_progress,
            iter_sampling=n_iter,
            chains=n_chains,
            inits=inits,
            seed=0,
            show_console=True
        )
    )
    for i in tqdm(range(1,n_repeat)):
        fit = az.from_cmdstanpy(model.sample(
            data=data,
            show_progress=show_progress,
            iter_sampling=n_iter,
            chains=n_chains,
            seed=i,
            inits=inits,
            show_console=True
        ))

        idata = az.concat(idata, fit, dim="chain")
    
 
    idata.to_netcdf(output_file_name)

    with open(output_file_name_time, 'w') as f:
        f.write(str(time.time() - start_time))

    if return_idata==True:
        return idata