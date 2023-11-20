#!/bin/bash
#SBATCH --partition=genx --job-name="sample"
source /mnt/home/mjhajaria/miniconda3/etc/profile.d/conda.sh
conda activate stan
cd /mnt/home/mjhajaria/transforms

if [ "$#" -ne 2 ]; then  # Changed this line to expect 2 arguments
    echo "Usage: $0 <transform> <target>"
    exit 1
fi

transform="$1"
target="$2"
echo $transform
echo $target
python3 /mnt/home/mjhajaria/transforms/utils/sample-script.py --transform="$transform" --target_keyword="$target"
