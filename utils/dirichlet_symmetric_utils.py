import numpy as np
from utils import sample, retrieve, cumulative_mean, rmse, rmse_leapfrog

def get_dirichlet_symmetric_params():
    alphas=[0.1,1,10]
    Ks = [10,100,1000]
    parameters = []
    for alpha in alphas:
        for K in Ks:
            parameters.append({'alpha': [alpha]*K, 'K': K})
            
    return parameters

def get_dirichlet_symmetric_rmse(transform, parameters, n_repeat=1):
    
    idata = retrieve(transform_category="simplex", transform=transform, n_repeat=n_repeat,
                 evaluating_model="dirichlet_symmetric", parameters=parameters)
    
    alpha = np.asarray(parameters['alpha'])
    true_var = alpha/sum(alpha)
    var_name='x'
    var_dim=0
    
    return rmse_leapfrog(idata, true_var, var_name, var_dim) 