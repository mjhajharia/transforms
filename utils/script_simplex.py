from utils import sample, retrieve, cumulative_mean, rmse
from dirichlet_symmetric_utils import get_dirichlet_symmetric_rmse, get_dirichlet_symmetric_params
import pickle
import numpy as np
import matplotlib.pyplot as plt

import os
os.chdir('..')

transforms = ['stan', 'stickbreaking', 'softmax', 'softmax-augmented']
for transform in transforms:
    sample(transform_category="simplex", transform=transform, evaluating_model="dirichlet_symmetric", 
           parameters={}, auto_eval_all_params=True, n_iter = 1000, n_chains = 4, n_repeat = 100)