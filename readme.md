# Transforms

# Directory Structure
```
├── example.ipynb
├── figures
│   ├── ess_arviz_example.png
│   ├── ess_matplotlib_example.png
│   └── rmse_example.png
├── readme.md
├── sampling_results
│   └── simplex
│       ├── softmax
│       ├── stan
│       └── stickbreaking
├── stan_models
│   ├── softmax-augmented_dirichlet_symmetric.stan
│   ├── softmax_dirichlet_symmetric.stan
│   ├── stan_dirichlet_symmetric.stan
│   └── stickbreaking_dirichlet_symmetric.stan
├── target_densities
│   ├── dirichlet_symmetric.stan
│   ├── dirichlet_symmetric_parameters.json
│   └── param_map_dirichlet_symmetric.json
├── tex
│   ├── all.bib
│   ├── makefile
│   └── paper.tex
├── transforms
│   ├── bounded
│   │   ├── constraint-stan
│   │   ├── constraint-stan.stan
│   │   ├── exp_transform
│   │   └── exp_transform.stan
│   └── simplex
│       ├── softmax-augmented.stan
│       ├── softmax.stan
│       ├── stan_simplex.stan
│       └── stickbreaking.stan
└── utils
    ├── dirichlet_symmetric.py
    ├── ess.py
    ├── rhat.py
    ├── rmse.py
    ├── sample.py
    └── script_simplex.py

```
