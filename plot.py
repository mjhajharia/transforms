
import os
os.chdir('transforms')

import sys
sys.path.insert(1, 'utils')
from sample import sample


import numpy as np
import matplotlib.pyplot as plt
import arviz as az

from dirichlet_symmetric import get_dirichlet_symmetric_rmse
from rhat import calc_rhat_mixed_chains
from ess import get_ess_leapfrog_ratio

transform_category='simplex'
evaluating_model='dirichlet_symmetric'

parameters = [{'alpha': [0.1]*10, 'N': 10}, {'alpha': [0.1]*100, 'N': 100}, {'alpha': [0.1]*1000, 'N': 1000},
{'alpha': [1]*10, 'N': 10}, {'alpha': [1]*100, 'N': 100}, {'alpha': [1]*1000, 'N': 1000},
{'alpha': [1]*10, 'N': 10}, {'alpha': [1]*100, 'N': 100}, {'alpha': [1]*1000, 'N': 1000}]

transforms = ['softmax', 'softmax-augmented','stickbreaking', 'stan']

get_dirichlet_symmetric_rmse(transforms, transform_category, 
                             parameters, fig_name='rmse_example.png')
