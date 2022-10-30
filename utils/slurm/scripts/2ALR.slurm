#!/bin/bash
#SBATCH --partition=ccm -N1 --job-name='2ALR'
source ~/miniconda3/etc/profile.d/conda.sh
conda activate stan
 cd ~/transforms
 python3 utils/script-simplex.py --parameters=2 --transform='ALR'
