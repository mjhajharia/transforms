#!/usr/bin/env python
# coding: utf-8

from cmdstanpy import CmdStanModel
import arviz as az
import os
import numpy as np
from tqdm import tqdm
import logging
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
os.chdir('..')
logger = logging.getLogger('cmdstanpy')
logger.addHandler(logging.NullHandler())

def get_ess_leapfrog_plot(transforms=['stickbreaking', 'softmax', 'softmax-augmented'], 
                          params={'alpha':[0.1]*10,'N':10}, 
                          var_name='x', var_dim=0, repeat=100, plot_format='matplotlib', fig_name='figure.png'):
   
    def get_ess_leapfrog_ratio(transform, params={'alpha':[0.1]*10,'N':10}, 
                          var_name='x', var_dim=0, repeat=100):
        model = CmdStanModel(stan_file = f'transforms/simplex/{transform}/dirichlet_symmetric.stan', 
                         cpp_options = {'STAN_THREADS':'true'})
        x=[]
        for i in tqdm(range(repeat)):
            idata = az.from_cmdstanpy(model.sample(data=params,show_progress=False ))
            data=idata.sel(var_names=[str(var_name)], x_dim_0=var_dim)
            ess = az.ess(data, var_names=[str(var_name)], method="bulk")[str(var_name)].values
            leapfrog = idata.sample_stats['n_steps'].sum().values
            x.append(ess/leapfrog)
        return x

    data={}
    for transform in transforms:
        data[str(transform)] = get_ess_leapfrog_ratio(transform,repeat=100)
        
    if plot_format=='matplotlib':
        fig, ax = plt.subplots()
        for transform in transforms:
            kde = gaussian_kde(data[str(transform)])
            dist_space = np.linspace( min(data[str(transform)]), max(data[str(transform)]), 1000 )
            ax.plot( dist_space, kde(dist_space), label=transform)
        ax.set_title(str(params))
        ax.axes.yaxis.set_ticklabels([])
        plt.xlabel('ESS/Leapfrog')
        plt.ylabel('Density')
        plt.legend()
        plt.savefig(str(fig_name), dpi=300)
        plt.show()

    if plot_format=='arviz':
        softmax= az.convert_to_inference_data(np.asarray(data['softmax']))
        stickbreaking= az.convert_to_inference_data(np.asarray(data['stickbreaking']))
        softmax_augmented= az.convert_to_inference_data(np.asarray(data['softmax-augmented']))
        az.plot_density([stickbreaking, softmax, softmax_augmented], shade=0.5, figsize=(10,5),hdi_prob=0.96,
                    data_labels=['stickbreaking', 'softmax', 'softmax-augmented']);

