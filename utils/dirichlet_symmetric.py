import numpy as np
import pickle
import matplotlib.pyplot as plt
from rmse import rmse_leapfrog
from sample import sample

def get_dirichlet_symmetric_rmse(transforms, transform_category, parameters, fig_name,  n_iter=1000, 
                                n_chains=4, n_repeat=1, show_progress=True, resample=False, output_dir='/mnt/sdceph/users/mjhajaria/'):
    
    plt.rcParams["figure.figsize"] = [20,10]
    plt.rcParams['figure.dpi'] = 300
    fig, ax = plt.subplots(1)
    fig.supxlabel('Cumulative Leapfrog Steps')
    fig.supylabel('Root Mean Squared Error')

    for transform in transforms:            
        idata = sample(transform_category=transform_category, transform=transform, 
            evaluating_model='dirichlet_symmetric', parameters=parameters, 
            auto_eval_all_params=False, n_iter = n_iter,  n_chains = n_chains, n_repeat=n_repeat,
                                    show_progress = show_progress, resample=resample, output_dir=output_dir,return_idata=True)
        alpha = parameters[0]['alpha']
        N = parameters[0]['N']
        true_x = [a/sum(alpha) for a in alpha]
        x, y = rmse_leapfrog(idata=idata, true_var=true_x, var_name='x', var_dim=0)
        ax.plot(x,y, label = str(transform))
        ax.legend()
        ax.set_title(f'alpha={alpha[0]}, N = {N}')
    fig.legend(labels=transforms,bbox_to_anchor = (0.6, -0.05));
    fig.savefig(f'figures/{fig_name}', dpi=300)

def create_param_map():
    alphas = [0.1, 1, 10]
    Ns = [10, 100, 1000]
    keys = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    parameters = []
    for alpha in alphas:
        for N in Ns:
            parameters.append({"alpha": [alpha]*N, "N": N})

    param_map = dict(zip(keys, parameters))
    param_map.update(dict(zip(parameters, keys)))

    pickle.dump(param_map, open("param_map_dirichlet_symmmetric.pkl", "wb"))

def get_dirichlet_symmetric_params():
    return pickle.load(open("param_map_dirichlet_symmmetric.pkl", "rb"))

