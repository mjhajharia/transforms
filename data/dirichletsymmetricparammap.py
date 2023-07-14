import json

alphas = [0.1, 1, 10]
Ns = [10, 100, 1000]
keys = [1, 2, 3, 4, 5, 6, 7, 8, 9]
parameters = []
for alpha in alphas:
    for N in Ns:
        parameters.append(tuple([alpha]*N))

param_map = dict(zip(keys, parameters))
with open('data/dirichletsymmetric.json', 'w') as fp:
    json.dump(param_map, fp)