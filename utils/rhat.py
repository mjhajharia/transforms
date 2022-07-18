import os
import arviz as az
import numpy as np
from cmdstanpy import CmdStanModel
import logging

logger = logging.getLogger("cmdstanpy")
logger.addHandler(logging.NullHandler())


def calc_rhat_mixed_chains(path_1, path_2, variable, data, force_compile=False):

    """
    This function evaluates rhat for 4 chains from 2 different models, as in 2 chains from each.

    Parameters
    ----------

    path_1: str
        path for the first model

    path_2: str
        path for the second model

    variable: str
        Variable to evaluate rhat for

    data: dict
        Dictionary of data

    force_compile: Bool
        Whether to recompile or not

    """
    # Build stan model
    file_1 = os.path.join(path_1)
    file_2 = os.path.join(path_2)
    model_1 = CmdStanModel(stan_file=file_1, cpp_options={"STAN_THREADS": "true"})
    model_2 = CmdStanModel(stan_file=file_2, cpp_options={"STAN_THREADS": "true"})

    # Recompile
    if force_compile is True:
        model_1.compile(force=True)
        model_2.compile(force=True)

    rhat_array = []
    for i in range(10):
        # Fit stan model
        fit_1 = model_1.sample(
            data=data, show_progress=False, iter_sampling=1000, chains=4
        )
        fit_2 = model_2.sample(
            data=data, show_progress=False, iter_sampling=1000, chains=4
        )

        # Convert to idata for arviz
        idata_1 = az.from_cmdstanpy(fit_1)
        idata_2 = az.from_cmdstanpy(fit_2)

        # Stack the samples according to chains
        stacked_1 = az.extract_dataset(idata_1)
        stacked_2 = az.extract_dataset(idata_2)

        # Concatenate chains and evaluate rhat
        chains = np.concatenate(
            (
                stacked_1.sel(chain=1)[variable],
                stacked_1.sel(chain=2)[variable],
                stacked_2.sel(chain=1)[variable],
                stacked_2.sel(chain=2)[variable],
            ),
            axis=1,
        )
        rhat = az.rhat(chains, var_names=variable, method="rank")
        rhat_array.append(rhat)

    return np.asarray(rhat_array).mean()
