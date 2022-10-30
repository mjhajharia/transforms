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


def get_rmse_plot(transform_category, evaluating_model):

    transforms = list_transforms(transform_category)[:1]
    transform_label = transforms_labels(transform_category)
    parameters = [list_params(evaluating_model)[0], list_params(evaluating_model)[1]]

    plt.rcParams["figure.figsize"] = [20,10]
    fig, axes = plt.subplots(3,3)

    for ax, params in zip(axes.flatten() if len(parameters)>1 else [axes],  parameters):
        for transform in tqdm(transforms):

            idata = sample(transform_category=transform_category, transform=transform,
            evaluating_model=evaluating_model, parameters=[params],
            auto_eval_all_params=False, n_iter = 1000,  n_chains = 4, n_repeat=2,
            show_progress = True, resample=True,return_idata=True, output_dir='')

            true_x, title = get_true_x(params,evaluating_model)
            x, y = rmse_leapfrog(idata=idata, true_var=true_x, var_name='x', var_dim=0)
            ax.set_title(str(title))
            ax.plot(x,y, label=transform_label[str(transform)])
            ax.legend()
            print(transform_label[str(transform)])

    ax.axes.yaxis.set_ticklabels([])
    fig.supxlabel('Cumulative Leapfrog Steps')
    fig.supylabel('Root Mean Squared Error')
    plt.savefig(f'figures/{transform_category}/rmse.png', dpi=300)

        

