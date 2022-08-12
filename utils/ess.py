from scipy.stats import gaussian_kde
import numpy as np
import matplotlib.pyplot as plt
from cmdstanpy import CmdStanModel
from tqdm import tqdm
import arviz as az
import sys
sys.path.insert(1, 'utils')
from sample import sample
import pickle

def get_ess_leapfrog_ratio(
    transform_category,
    transform,
    evaluating_model,
    params,
    var_name,
    var_dim,
    n_repeat=100,
    plot_type='density'
):
    x = []
    idata = sample(
        transform_category=transform_category,
        transform=transform,
        evaluating_model=evaluating_model,
        parameters=[params],
        output_dir='/mnt/sdceph/users/mjhajaria',
        auto_eval_all_params=False,
        n_iter=1000,
        n_chains=4,
        n_repeat=100,
        show_progress=True,
        resample=False,
        return_idata=True)
    
    with open(f"target_densities/param_map_{evaluating_model}.pkl", "rb") as f:
        param_map = pickle.load(f)
    ess = np.loadtxt(open(f'/mnt/sdceph/users/mjhajaria/sampling_results/{transform_category}/{transform}/{evaluating_model}/ess_{param_map[tuple(list(params.values())[0])]}_{n_repeat}.csv'),delimiter = ",")
    leapfrog = np.average(idata.sample_stats['n_steps'].sum(axis=1).values.reshape(-1, 4), axis=1)
    x=np.divide(ess, leapfrog)
    if plot_type == 'density':
    	kde = gaussian_kde(x)
    	dist_space = np.linspace(min(x), max(x), 10000)
    	return dist_space, kde(dist_space)
    if plot_type=='cdf':
        count, bins_count = np.histogram(x, bins=100)
        pdf = count / sum(count)
        cdf = np.cumsum(pdf)
        return bins_count[1:], cdf

