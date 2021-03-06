
import sys
sys.path.insert(1, 'utils')
import numpy as np
import matplotlib.pyplot as plt
import arviz as az

from ess import get_ess_leapfrog_ratio


# parameters = [{'alpha':[1]*10, 'N':10}, {'alpha':[1]*100, 'N':100}, {'alpha':[1]*1000, 'N':1000},
#               {'alpha':[1]*10, 'N':10}, {'alpha':[1]*100, 'N':100},{'alpha':[1]*1000, 'N':1000},
#               {'alpha':[10]*10, 'N':10}, {'alpha':[10]*100, 'N':100},{'alpha':[10]*1000, 'N':1000}]

parameters = [{'alpha':[1]*10, 'N':10}]
# transforms = ['stickbreaking', 'softmax', 'softmax-augmented', 'stan']
transforms = ['stickbreaking']
transform_category='simplex'
evaluating_model='dirichlet_symmetric'

var_name='x'
var_dim=0

import pickle

fig, axes = plt.subplots(1)
for ax, params in zip(axes.flatten() if len(parameters)>1 else [axes],  parameters):
    for transform in transforms:
        x, y = get_ess_leapfrog_ratio(transform_category, transform, evaluating_model, params, var_name, var_dim, n_repeat=100)
        ax.plot(x,y, label=transform)
    ax.set_title(f'alpha = {params["alpha"]}, N = {params["N"]}')
ax.axes.yaxis.set_ticklabels([])
plt.xlabel('ESS/Leapfrog')
plt.ylabel('Density')
plt.legend()
plt.savefig('figures/ess_experimental.png', dpi=300)
