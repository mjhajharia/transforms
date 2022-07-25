

import os
os.getcwd()

from sample import sample

import arviz as az

x = []
idata = sample(
    transform_category='simplex',
    transform='softmax',
    evaluating_model='dirichlet_symmetric',
    parameters=[{'alpha':[0.1]*10, 'N':10}],
    output_dir='/mnt/sdceph/users/mjhajaria',
    auto_eval_all_params=False,
    n_iter=1000,
    n_chains=4,
    show_progress=True,
    resample=False,
    return_idata=True)
for i in tqdm(range(100)):
    x_data = idata.sel(chains=[i],var_names=[x], x_dim_0=0)
    ess = az.ess(x_data)['x'].value
    print(ess)
    leapfrog = x_data.sample_stats["n_steps"].sum().values
    x.append(ess / leapfrog)
print(x)
