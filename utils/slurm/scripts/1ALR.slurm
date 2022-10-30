#!/bin/bash
#SBATCH --partition=ccm -N1 --job-name='1ALR'
source ~/miniconda3/etc/profile.d/conda.sh
conda activate stan
 cd ~/transforms
 python3 utils/script-simplex.py --parameters=1 --transform='ALR'
