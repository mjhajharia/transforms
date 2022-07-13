from scipy.stats import gaussian_kde
import numpy as np
import matplotlib.pyplot as plt
from cmdstanpy import CmdStanModel
from tqdm import tqdm
import arviz as az

def get_ess_leapfrog_ratio(transform_category, transform, evaluating_model, params, var_name, var_dim, repeat=100, return_ess=False):
    model = CmdStanModel(stan_file = f'stan_models/{transform}_{evaluating_model}.stan', cpp_options = {'STAN_THREADS':'true'})
    x=[]
    for i in tqdm(range(repeat)):
        idata = az.from_cmdstanpy(model.sample(data=params,show_progress=False ))
        data=idata.sel(var_names=[str(var_name)], x_dim_0=var_dim)
        ess = az.ess(data, var_names=[str(var_name)], method="bulk")[str(var_name)].values
        leapfrog = idata.sample_stats['n_steps'].sum().values
        x.append(ess/leapfrog)
    if return_ess==True:
        return x
    else:
        kde = gaussian_kde(x)
        dist_space = np.linspace( min(x), max(x), 1000)
        return dist_space, kde(dist_space)



