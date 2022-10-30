import numpy as np
import pickle
import matplotlib.pyplot as plt
from sample import sample


def list_transforms(transform_category='simplex'):
    if transform_category=='simplex':
        return ['Stickbreaking', 'ALR'
        'AugmentedSoftmax', 'StanStickbreaking','AugmentedILR', 
        'Hyperspherical', 'HypersphericalAngular', 'HypersphericalLogit','LogisticProduct']

def transforms_labels(transform_category='simplex'):
    if transform_category=='simplex':
        labels={'Stickbreaking': 'Stick-breaking',
                        'ALR': 'Additive Log Ratio',
                        'AugmentedSoftmax': 'Augmented-Softmax',
                        'StanStickbreaking': 'Stick-breaking (in C++)',
                        'AugmentedILR'  : 'Augmented-Isometric Log Ratio',
                        'Hyperspherical': 'Hyperspherical',
                        'HypersphericalAngular': 'Hyperspherical-Angular',
                        'HypersphericalLogit': 'Hyperspherical-Logit',
                        'LogisticProduct': 'Logistic-Product',
                        }
        return labels

def list_params(evaluating_model='DirichletSymmetric'):
    if evaluating_model=='DirichletSymmetric':
        parameters = [{'alpha':[0.1]*10, 'N':10}, {'alpha':[0.1]*100, 'N':100}, {'alpha': [0.1]*1000, 'N': 1000},
                    {'alpha':[1]*10, 'N':10}, {'alpha':[1]*100, 'N':100},  {'alpha': [1]*1000, 'N': 1000},
                    {'alpha':[10]*10, 'N':10}, {'alpha':[10]*100, 'N':100},  {'alpha': [10]*1000, 'N': 1000}]
        return parameters
def get_true_x(params, evaluating_model='DirichletSymmetric'):
    if evaluating_model=='DirichletSymmetric':
        alpha = params['alpha']
        N = params['N']
        true_x = [a/sum(alpha) for a in alpha]
        title = f'alpha = {params["alpha"][0]}, N = {params["N"]}'
        return true_x, title

def get_DirichletSymmetric_rmse(transforms, transform_category, parameters, fig_name,  n_iter=1000, 
                                n_chains=4, n_repeat=1, show_progress=True, resample=False, output_dir='/mnt/sdceph/users/mjhajaria/'):
    
    plt.rcParams["figure.figsize"] = [20,10]
    plt.rcParams['figure.dpi'] = 300
    fig, ax = plt.subplots(1)
    fig.supxlabel('Cumulative Leapfrog Steps')
    fig.supylabel('Root Mean Squared Error')

    for transform in transforms:            
        idata = sample(transform_category=transform_category, transform=transform, 
            evaluating_model='DirichletSymmetric', parameters=parameters, 
            auto_eval_all_params=False, n_iter = n_iter,  n_chains = n_chains, n_repeat=n_repeat,
                                    show_progress = show_progress, resample=resample, output_dir=output_dir,return_idata=True)
        alpha = parameters[0]['alpha']
        N = parameters[0]['N']
        true_x = [a/sum(alpha) for a in alpha]
        x, y = rmse_leapfrog(idata=idata, true_var=true_x, var_name='x', var_dim=0)
        ax.plot(x,y, label = str(transform))
        ax.legend()
        ax.set_title(f'alpha={alpha[0]}, N = {N}')
    fig.legend(labels=transforms,bbox_to_anchor = (0.6, -0.05))
    fig.savefig(f'figures/{fig_name}', dpi=300)

def create_param_map():
    alphas = [0.1, 1, 10]
    Ns = [10, 100, 1000]
    keys = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    parameters = []
    for alpha in alphas:
        for N in Ns:
            parameters.append(tuple([alpha]*N))

    param_map = dict(zip(keys, parameters))
    reverse_map = dict(zip(parameters, keys))
    param_map.update(reverse_map)

    pickle.dump(param_map, open("target_densities/param_map_DirichletSymmetric.pkl", "wb"))

    with open(f"target_densities/param_map_DirichletSymmetric.pkl", "rb") as f:
        param_map = pickle.load(f)
    return param_map