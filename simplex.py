import os
from tqdm import tqdm
os.chdir('transforms')

import sys
sys.path.insert(1, 'utils')

from sample import sample

parameters = [{'alpha': [0.1]*10, 'N': 10}, {'alpha': [0.1]*100, 'N': 100}, {'alpha': [0.1]*1000, 'N': 1000},
{'alpha': [1]*10, 'N': 10}, {'alpha': [1]*100, 'N': 100}, {'alpha': [1]*1000, 'N': 1000},
{'alpha': [1]*10, 'N': 10}, {'alpha': [1]*100, 'N': 100}, {'alpha': [1]*1000, 'N': 1000}]

for transform in tqdm(transforms):
    for params in tqdm(parameters):
        sample(transform_category='simplex', transform=transforms, 
            evaluating_model='dirichlet_symmetric', parameters=[params], 
            auto_eval_all_params=False, n_iter = 1000, n_chains = 4, n_repeat = 100, 
                            show_progress = True, resample=True, return_idata=False)


