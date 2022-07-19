import os
from tqdm import tqdm
os.chdir('transforms')

import sys
sys.path.insert(1, 'utils')

from sample import sample

transforms=['stan', 'stickbreaking', 'softmax', 'softmax-augmented']

for transform in tqdm(transforms):
        sample(transform_category='simplex', transform=transform, 
            evaluating_model='dirichlet_symmetric', parameters={}, 
            auto_eval_all_params=True, n_iter = 1000, n_chains = 4, n_repeat = 1, 
                            show_progress = True, resample=True, return_idata=False)


