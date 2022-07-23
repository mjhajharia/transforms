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
        transform_category=transform_category,
        transform=transform,
        evaluating_model=evaluating_model,
        parameters=[params],
        output_dir='/mnt/sdceph/users/mjhajaria',
        auto_eval_all_params=False,
        n_iter=1000,
        n_chains=4,
        show_progress=True,
        resample=False,
        return_idata=True)
    for i in tqdm(range(n_repeat)):
        x_data = idata.sel(chains=[i],var_names=[str(var_name)], x_dim_0=var_dim)
        ess = az.ess(x_data)['x'].value
        print(ess)
        leapfrog = x_data.sample_stats["n_steps"].sum().values
        x.append(ess / leapfrog)
    print(x)
    kde = gaussian_kde(x)
    dist_space = np.linspace(min(x), max(x), 1000)
    print(dist_space)
    return dist_space, kde(dist_space)
