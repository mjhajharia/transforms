from scipy.stats import gaussian_kde
import numpy as np
import matplotlib.pyplot as plt
from cmdstanpy import CmdStanModel
from tqdm import tqdm
import arviz as az
import sys
sys.path.insert(1, 'utils')
from sample import sample

def get_ess_leapfrog_ratio(
    transform_category,
    transform,
    evaluating_model,
    params,
    var_name,
    var_dim,
    n_repeat=100,
):
    x = []
    idata = sample(
        transform_category='simplex',
        transform='softmax',
        evaluating_model='dirichlet_symmetric',
        parameters=[{'alpha': [0.1]*10, 'N': 10}],
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

    ess = np.loadtxt(open('/mnt/sdceph/users/mjhajaria/sampling_results/simplex/softmax/dirichlet_symmetric/ess_{param_map[tuple(list(params.values())[0])]}_{n_repeat}.csv'),delimiter = ",")
    leapfrog = np.average(idata.sample_stats['n_steps'].sum(axis=1).reshape(-1, 4), axis=1)
    x=np.divide(ess, leapfrog)
    kde = gaussian_kde(x)
    dist_space = np.linspace(min(x), max(x), 1000)
    return dist_space, kde(dist_space)


