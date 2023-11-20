#!/bin/bash
for k in 'dirsym_10_100dims' 
# 'dirsym_e-1_10dims' 'dirsym_e-1_100dims' 'dirsym_e-1_1000dims' 'dirsym_1_10dims' 'dirsym_1_100dims' 'dirsym_1_1000dims' 'dirsym_10_10dims' 'dirsym_10_100dims' 'dirsym_10_1000dims' 'asymdir_1toe-1_10dims' 'asymdir_e-1to1_10dims' 'asymdir_1to10_10dims' 'asymdir_e-1to10_100dims' 'asymdir_e-1to10_1000dims' 'asymdir_e-1rand1_10dims' 'asymdir_1rand10_10dims' 'asymdir_e-1rand10_100dims' 'asymdir_e-1rand10_1000dims' 'logitnormal_rho5e-1_scaleunit_10dims' 'logitnormal_rho5e-1_scalevar_10dims' 'logitnormal_rho5e-1_scalevar_reverse_10dims' 'logitnormal_rho95e-2_scaleunit_10dims' 'logitnormal_rho95e-2_scalevar_10dims' 'logitnormal_rho5e-1_scaleunit_100dims' 'logitnormal_rho5e-1_scalevar_100dims' 'logitnormal_rho95e-2_scaleunit_100dims' 'logitnormal_rho95e-2_scalevar_100dims'
do
    for j in 'ALR' 'HypersphericalProbit' 'NormalizedExponential' 'StanStickbreaking' 'HypersphericalAngular' 'AugmentedILR'
    # 'Stickbreaking' 'ALR' 'NormalizedExponential' 'AugmentedILR' 'HypersphericalAngular' 'HypersphericalLogit' 'StanStickbreaking' 'HypersphericalProbit' 'ProbitProduct' 'AugmentedSoftmax'
    do
        sbatch -c1 utils/sample.sh $j $k
    done
done