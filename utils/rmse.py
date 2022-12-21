from sample import sample

import numpy as np
import matplotlib.pyplot as plt
import arviz as az
from tqdm import tqdm
from utils import list_transforms, get_true_x, transforms_labels, list_params

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def cumulative_mean(x):
    return np.divide(np.cumsum(x), np.arange(1, len(x) + 1))


def rmse_leapfrog(idata, true_var, var_name, var_dim):
    cumulative_leapfrog_steps = np.cumsum(np.mean(idata.sample_stats.n_steps, axis=0))
    pred_var = cumulative_mean(np.mean(idata.posterior[str(var_name)].sel(x_dim_0=var_dim), axis=0))
    rmse_array = []
    for i in range(1, len(pred_var) + 1):
        rmse_array.append(rmse(true_var[var_dim], pred_var[:i]))
    return cumulative_leapfrog_steps, rmse_array


def get_rmse_plot(transform_category, evaluating_model, transforms=None, parameters=None, subplot_x=3, subplot_y=3, output_dir='/mnt/home/mjhajaria/ceph/', n_repeat=100):

    if transforms is None:
        transforms = list_transforms(transform_category)
    
    if parameters is None:
        parameters = list_params(evaluating_model)
    
    transform_label = transforms_labels(transform_category)
    plt.rcParams["figure.figsize"] = [20,10]
    fig, axes = plt.subplots(subplot_x,subplot_y)

    for ax, params in zip(axes.flatten() if len(parameters)>1 else [axes],  parameters):
        for transform in tqdm(transforms):
            try:

                idata = sample(transform_category=transform_category, transform=transform,
                evaluating_model=evaluating_model, parameters=[params],
                auto_eval_all_params=False, n_iter = 1000,  n_chains = 4, n_repeat=n_repeat,
                show_progress = True, resample=False,return_idata=True, output_dir=output_dir)

                true_x, title = get_true_x(params,evaluating_model)
                x, y = rmse_leapfrog(idata=idata, true_var=true_x, var_name='x', var_dim=0)
                ax.set_title(str(title))
                ax.plot(x,y, label=transform_label[str(transform)])
                print(transform_label[str(transform)])

            except FileNotFoundError:
                print(f' no sampler data for parametrization {params} and transform {transform_label[str(transform)]}')
            
    handles, labels = fig.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(),loc='upper center', bbox_to_anchor=(0.5, 0.97),
    ncol=4, fancybox=True, shadow=True)
            

    ax.axes.yaxis.set_ticklabels([])
    fig.supxlabel('Cumulative Leapfrog Steps')
    fig.supylabel('Root Mean Squared Error')
    plt.savefig(f'figures/{transform_category}/rmse.png', dpi=300)

        

