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

from utils import list_params, list_transforms, transforms_labels

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
        output_dir='',
        auto_eval_all_params=False,
        n_iter=1000,
        n_chains=4,
        n_repeat=100,
        show_progress=True,
        resample=True,
        return_idata=True)
    
    with open(f"target_densities/param_map_{evaluating_model}.pkl", "rb") as f:
        param_map = pickle.load(f)
    ess = np.loadtxt(open(f'sampling_results/{transform_category}/{transform}/{evaluating_model}/ess_{param_map[tuple(list(params.values())[0])]}_{n_repeat}.csv'),delimiter = ",")
    leapfrog = np.average(idata.sample_stats['n_steps'].sum(axis=1).values.reshape(-1, 4), axis=1)
    
    x=np.divide(ess, leapfrog)

    if plot_type == 'density':
    	kde = gaussian_kde(x)
    	dist_space = np.linspace(min(x), max(x), 1000)
    	return dist_space, kde(dist_space)

    if plot_type=='cdf':
        count, bins_count = np.histogram(x, bins=10)
        pdf = count / sum(count)
        cdf = np.cumsum(pdf)
        return bins_count[1:], cdf



def get_ess_plot(plottype, transform_category, evaluating_model, var_name, var_dim):

    transforms = list_transforms(transform_category)
    transform_label = transforms_labels(transform_category)
    parameters = list_params(evaluating_model)

    plt.rcParams["figure.figsize"] = (20,10)
    fig, axes = plt.subplots(3,3)
    for ax, params in zip(axes.flatten() if len(parameters)>1 else [axes],  parameters):
        for transform in transforms:
            if plottype=='density':
                label='Probability Density Function'
                x, y = get_ess_leapfrog_ratio(transform_category, transform, evaluating_model, params, var_name, var_dim,  n_repeat=2, plot_type='density')
            if plottype=='cdf':
                label='Cumulative Probability'
                x, y = get_ess_leapfrog_ratio(transform_category, transform, evaluating_model, params, var_name, var_dim, n_repeat=2, plot_type='cdf')
            ax.plot(x,y, label=transform_label[str(transform)])
        ax.set_title(f'alpha = {params["alpha"][0]}, N = {params["N"]}')
    ax.axes.yaxis.set_ticklabels([])
    fig.supxlabel('ESS/Leapfrog')
    fig.supylabel(str(label))
    plt.legend()
    plt.savefig(f'figures/simplex/ess_{plottype}.png', dpi=300)
