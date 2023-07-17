#!/bin/bash
#SBATCH --partition=genx  --job-name="AugmentedILR"
source /mnt/home/mjhajaria/miniconda3/etc/profile.d/conda.sh
conda activate stan
cd /mnt/home/mjhajaria/transforms
# for i in 1 2 3 4 5 6 7 8 9
# do
python3 /mnt/home/mjhajaria/transforms/utils/simplex-scripts/script-simplex.py --transform="ProbitProduct" --datakey="8"
# done