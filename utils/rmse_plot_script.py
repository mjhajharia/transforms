import os
os.chdir('..')

import sys
sys.path.insert(1, 'utils')
from sample import sample
from rmse import rmse_leapfrog

import numpy as np
import matplotlib.pyplot as plt
import arviz as az

transform_category='simplex'
evaluating_model='dirichlet_symmetric'

transforms = ['stickbreaking', 'softmax', 'softmax-augmented', 'stan']

transform_label = {'stickbreaking': 'Stick-breaking',
                   'softmax': 'Additive Log Ratio',
                   'softmax-augmented': 'Augmented Softmax',
                   'stan': 'Stick-breaking (in C++)'}

parameters = [{'alpha':[0.1]*10, 'N':10}, {'alpha':[0.1]*100, 'N':100}, {'alpha': [0.1]*1000, 'N': 1000},
               {'alpha':[1]*10, 'N':10}, {'alpha':[1]*100, 'N':100},  {'alpha': [1]*1000, 'N': 1000},
               {'alpha':[10]*10, 'N':10}, {'alpha':[10]*100, 'N':100},  {'alpha': [10]*1000, 'N': 1000}]

plt.rcParams["figure.figsize"] = [20,10]
fig, axes = plt.subplots(3,3)
fig.supxlabel('Cumulative Leapfrog Steps')
fig.supylabel('Root Mean Squared Error')

for ax, params in zip(axes.flatten() if len(parameters)>1 else [axes],  parameters):
    for transform in transforms:
        idata = sample(transform_category=transform_category, transform=transform,
            evaluating_model='dirichlet_symmetric', parameters=[params],
            auto_eval_all_params=False, n_iter = 1000,  n_chains = 4, n_repeat=100,
                                    show_progress = True, resample=False,return_idata=True)
        alpha = params['alpha']
        N = params['N']
        true_x = [a/sum(alpha) for a in alpha]
        x, y = rmse_leapfrog(idata=idata, true_var=true_x, var_name='x', var_dim=0)
        ax.plot(x,y, label=transform_label[str(transform)])
        print(transform_label[str(transform)])
    ax.set_title(f'alpha={alpha[0]}, N = {N}')
ax.axes.yaxis.set_ticklabels([])
plt.legend()
plt.savefig('figures/simplex/rmse.png', dpi=300)

    

