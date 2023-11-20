import pickle
import numpy as np
np.random.seed(10)

def ar_1_covariance(sigma, rho):
    N = len(sigma)
    covariance = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            correlation = rho ** abs(i - j)
            covariance[i, j] = (sigma[i] * sigma[j] * correlation)/(1-rho**2)
    return covariance

targets = {
    'dirsym_e-1_10dims': {'N': 10, "dirichlet_target": 1,
    "mu": [], "sigma": [[1]], "alpha": [0.1]*10},
    'dirsym_e-1_100dims': {'N': 100, "dirichlet_target": 1,
    "mu": [], "sigma": [[1]], "alpha": [0.1]*100},
    'dirsym_e-1_1000dims': {'N': 1000, "dirichlet_target": 1,
    "mu": [], "sigma": [[1]], "alpha": [0.1]*1000},
    'dirsym_1_10dims': {'N': 10, "dirichlet_target": 1,
    "mu": [], "sigma": [[1]], "alpha": [1]*10},
    'dirsym_1_100dims': {'N': 100, "dirichlet_target": 1,
    "mu": [], "sigma": [[1]], "alpha": [1]*100},
    'dirsym_1_1000dims': {'N': 1000, "dirichlet_target": 1,
    "mu": [], "sigma": [[1]], "alpha": [1]*1000},
    'dirsym_10_10dims': {'N': 10, "dirichlet_target": 1,
    "mu": [], "sigma": [[1]], "alpha": [10]*10},
    'dirsym_10_100dims': {'N': 100, "dirichlet_target": 1,
    "mu": [], "sigma": [[1]], "alpha": [10]*100},
    'dirsym_10_1000dims': {'N': 1000, "dirichlet_target": 1,
    "mu": [], "sigma": [[1]], "alpha": [10]*1000},


    'asymdir_1toe-1_10dims': {'N': 10, "dirichlet_target": 1,
    "mu": [], "sigma": [[1]], "alpha": np.linspace(1, 0.1, 10)},
    'asymdir_e-1to1_10dims': {'N': 10, "dirichlet_target": 1,
    "mu": [], "sigma": [[1]], "alpha": np.linspace(0.1, 1, 10)},
    'asymdir_1to10_10dims': {'N': 10, "dirichlet_target": 1,
    "mu": [], "sigma": [[1]], "alpha": np.linspace(1, 10, 10)},
    'asymdir_e-1to10_100dims': {'N': 100, "dirichlet_target": 1,
    "mu": [], "sigma": [[1]], "alpha": np.linspace(0.1, 10, 100)},
    'asymdir_e-1to10_1000dims': {'N': 1000, "dirichlet_target": 1,
    "mu": [], "sigma": [[1]], "alpha": np.linspace(0.1, 10, 1000)},
    'asymdir_e-1rand1_10dims': {'N': 10, "dirichlet_target": 1,
    "mu": [], "sigma": [[1]], "alpha": np.random.uniform(0.1, 1, 10)},
    'asymdir_1rand10_10dims': {'N': 10, "dirichlet_target": 1,
    "mu": [], "sigma": [[1]], "alpha": np.random.uniform(1, 10, 10)},
    'asymdir_e-1rand10_100dims': {'N': 100, "dirichlet_target": 1,
    "mu": [], "sigma": [[1]], "alpha": np.random.uniform(0.1, 10, 100)},
    'asymdir_e-1rand10_1000dims': {'N': 1000, "dirichlet_target": 1,
    "mu": [], "sigma": [[1]], "alpha": np.random.uniform(0.1, 10, 1000)},


    'logitnormal_rho5e-1_scaleunit_10dims': {'N': 10, "dirichlet_target": 0,
    "mu": [0]*(9), "sigma": ar_1_covariance(np.ones(9), 0.5),"alpha": []},
    'logitnormal_rho5e-1_scalevar_10dims': {'N': 10, "dirichlet_target": 0,
    "mu": [0]*(9), "sigma": ar_1_covariance(np.arange(1, 10, 1), 0.5),"alpha": []},
    'logitnormal_rho5e-1_scalevar_reverse_10dims': {'N': 10, "dirichlet_target": 0,
    "mu": [0]*(9), "sigma": ar_1_covariance(np.arange(10, 1, -1), 0.5),"alpha": []},
    'logitnormal_rho95e-2_scaleunit_10dims': {'N': 10, "dirichlet_target": 0,
    "mu": [0]*(9), "sigma": ar_1_covariance(np.ones(9), 0.95),"alpha": []},
    'logitnormal_rho95e-2_scalevar_10dims': {'N': 10, "dirichlet_target": 0,
    "mu": [0]*(9), "sigma": ar_1_covariance(np.arange(1, 10, 1), 0.95),"alpha": []},
    'logitnormal_rho5e-1_scaleunit_100dims': {'N': 100, "dirichlet_target": 0,
    "mu": [0]*(99), "sigma": ar_1_covariance(np.ones(99), 0.5),"alpha": []},
    'logitnormal_rho5e-1_scalevar_100dims': {'N': 100, "dirichlet_target": 0,
    "mu": [0]*(99), "sigma": ar_1_covariance(np.arange(1, 100, 1), 0.5),"alpha": []},
    'logitnormal_rho95e-2_scaleunit_100dims': {'N': 100, "dirichlet_target": 0,
    "mu": [0]*(99), "sigma": ar_1_covariance(np.ones(99), 0.95),"alpha": []},
    'logitnormal_rho95e-2_scalevar_100dims': {'N': 100, "dirichlet_target": 0,
    "mu": [0]*(99), "sigma": ar_1_covariance(np.arange(1, 100, 1), 0.95),"alpha": []}
}

for key, value in targets.items():
    if isinstance(value, dict):
        file_name = f'data/{key}.pickle'
        with open(file_name, 'wb') as handle:
            pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)

print(targets.keys())