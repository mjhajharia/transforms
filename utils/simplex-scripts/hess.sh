#!/bin/bash
#SBATCH --partition=genx  --job-name="hessians"
source /mnt/home/mjhajaria/miniconda3/etc/profile.d/conda.sh
conda activate stan
cd /mnt/home/mjhajaria/transforms
python3 /mnt/home/mjhajaria/transforms/utils/simplex-scripts/hess_fdm.py --transform="ALR" --datakey="3"