#!/bin/bash
#SBATCH --partition=genx  --job-name="summary"
source /mnt/home/mjhajaria/miniconda3/etc/profile.d/conda.sh
conda activate stan
cd /mnt/home/mjhajaria/transforms
python3 /mnt/home/mjhajaria/transforms/utils/simplex-scripts/summary-simplex.py 