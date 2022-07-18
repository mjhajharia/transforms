import os

os.chdir("transforms")

import sys

sys.path.insert(1, "utils")

from sample import sample

parameters = [{"alpha": [1] * 10, "N": 10}]
sample(
    transform_category="simplex",
    transform="stan",
    evaluating_model="dirichlet_symmetric",
    parameters=parameters,
    output_file="/mnt/ceph/users/mjhajaria/sampling_results/stan_1.json",
    auto_eval_all_params=False,
    n_iter=1000,
    n_chains=4,
    n_repeat=1,
    show_progress=True,
    resample=True,
    return_idata=False,
)
