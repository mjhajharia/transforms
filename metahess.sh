#!/bin/bash
for i in 'AugmentedILR' 'Stickbreaking' 'ALR''HypersphericalAngular' 'HypersphericalLogit' 'HypersphericalProbit' 'ProbitProduct' 'AugmentedSoftmax' 'NormalizedExponential' 'StanStickbreaking'
do
    for j in 1 2 4 5 7 8
    do
        sbatch -c1 hess.sh $i $j
    done

    for j in 3 6 9
    do
        sbatch -c2 hess.sh $i $j
    done
done