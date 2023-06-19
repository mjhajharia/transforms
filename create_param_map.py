import pickle

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