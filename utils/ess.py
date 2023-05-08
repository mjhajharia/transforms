from scipy.stats import gaussian_kde
import numpy as np
import matplotlib.pyplot as plt
from cmdstanpy import CmdStanModel
from tqdm import tqdm
import arviz as az
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
    plot_type='density'
):
    N = params['N']
    if N == 1000 and transform in ['Hyperspherical', 'HypersphericalAngular']:
        n_repeat=1
    else:
        n_repeat=100

    print(N, transform,n_repeat)
    idata = sample(
        transform_category=transform_category,
        transform=transform,
        evaluating_model=evaluating_model,
        parameters=[params],
        output_dir='/mnt/home/mjhajaria/ceph/',
        auto_eval_all_params=False,
        n_iter=1000,
        n_chains=4,
        n_repeat=n_repeat,
        show_progress=True,
        resample=False,
        return_idata=True)
    
    with open(f"target_densities/param_map_{evaluating_model}.pkl", "rb") as f:
        param_map = pickle.load(f)
    ess = np.loadtxt(open(f'/mnt/home/mjhajaria/transforms/ess_{transform}_{param_map[tuple(list(params.values())[0])]}.csv'),delimiter = ",")
    leapfrog = np.average(idata.sample_stats['n_steps'].sum(axis=1).values.reshape(-1, 4), axis=1)
    
    x=np.divide(ess, leapfrog)
    if len(x)==1:
            return np.full(100, 0), np.full(100,x[0])
    else:
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
            try:
                if plottype=='density':
                    label='Probability Density Function'
                    x, y = get_ess_leapfrog_ratio(transform_category, transform, evaluating_model, params, var_name, var_dim, plot_type='density')
                if plottype=='cdf':
                    label='Cumulative Probability'
                    x, y = get_ess_leapfrog_ratio(transform_category, transform, evaluating_model, params, var_name, var_dim, plot_type='cdf')
                
                ax.plot(x,y, label=transform_label[str(transform)])
            except FileNotFoundError:
                print(f' no sampler data for parametrization {params} and transform {transform_label[str(transform)]}')
            
        ax.set_title(f'alpha = {params["alpha"][0]}, N = {params["N"]}')
    ax.axes.yaxis.set_ticklabels([])
    fig.supxlabel('ESS/Leapfrog')
    fig.supylabel(str(label))
    handles, labels = fig.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(),loc='upper center', bbox_to_anchor=(0.5, 0.97),
          ncol=4, fancybox=True, shadow=True)
    plt.savefig(f'figures/simplex/ess_{plottype}.png', dpi=300)
