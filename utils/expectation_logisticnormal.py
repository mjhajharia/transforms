from cmdstanpy import CmdStanModel
import numpy as np

def softmax(y):
    return np.exp(y) / np.exp(y).sum(axis=0)

def additive_logistic_transformation(y):
    return softmax(np.insert(y, N-1, 0))

target_keyword='logitnormal_rho5e-1_scalevar_10dims'

with open(f'../data/{target_keyword}.pickle', 'rb') as file:
    data = pickle.load(file)

mu = [0.0]*(N-1)
np.random.seed(32)
sigma = data['sigma']
y = np.random.multivariate_normal(mu, sigma, size=10000)
xs=np.zeros((10000, N))
for i in range(10000):
     xs[i] = additive_logistic_transformation(y[i])
xs.mean(axis=0)