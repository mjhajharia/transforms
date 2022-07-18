import numpy as np
import matplotlib.pyplot as plt
from sample import sample


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def cumulative_mean(x):
    return np.divide(np.cumsum(x), np.arange(1, len(x) + 1))


def rmse_leapfrog(idata, true_var, var_name, var_dim):
    cumulative_leapfrog_steps = np.cumsum(np.mean(idata.sample_stats.n_steps, axis=0))
    pred_var = cumulative_mean(
        np.mean(idata.posterior[str(var_name)].sel(x_dim_0=var_dim), axis=0)
    )
    rmse_array = []
    for i in range(1, len(pred_var) + 1):
        rmse_array.append(rmse(true_var[var_dim], pred_var[:i]))
    return cumulative_leapfrog_steps, rmse_array
