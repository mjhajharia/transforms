# Transforms

# Directory Structure
```
├── cmdstan
├── figures
│   └── figure.png
├── readme.md
├── sampling_results
│   └── simplex
│       ├── softmax
│       │   └── dirichlet_symmetric
│       │       └── 4_1.json
│       ├── softmax-augmented
│       │   └── dirichlet_symmetric
│       │       └── 4_1.json
│       ├── stan
│       │   └── dirichlet_symmetric
│       │       └── 4_1.json
│       └── stickbreaking
│           └── dirichlet_symmetric
│               └── 4_1.json
├── tex
│   ├── all.bib
│   ├── makefile
│   ├── paper.pdf
│   └── paper.tex
├── transforms
│   ├── bounded
│   │   ├── constraint-stan
│   │   ├── constraint-stan.hpp
│   │   ├── constraint-stan.stan
│   │   ├── exp_transform
│   │   ├── exp_transform.hpp
│   │   ├── exp_transform.stan
│   │   └── whatever.ipynb
│   └── simplex
│       ├── dirichlet_symmetric_map.json
│       ├── dirichlet_symmetric_parameters.json
│       ├── softmax
│       │   ├── dirichlet_symmetric
│       │   ├── dirichlet_symmetric.hpp
│       │   └── dirichlet_symmetric.stan
│       ├── softmax-augmented
│       │   ├── dirichlet_symmetric
│       │   ├── dirichlet_symmetric.hpp
│       │   └── dirichlet_symmetric.stan
│       ├── stan
│       │   ├── dirichlet_symmetric
│       │   ├── dirichlet_symmetric.hpp
│       │   └── dirichlet_symmetric.stan
│       └── stickbreaking
│           ├── dirichlet_symmetric
│           ├── dirichlet_symmetric.hpp
│           └── dirichlet_symmetric.stan
└── utils
    ├── dirichlet_symmetric_utils.py
    ├── ess_plot.py
    ├── rhat_eval.py
    ├── rmse_plot.py
    ├── script.py
    └── utils.py
```
