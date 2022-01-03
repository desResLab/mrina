#!/bin/bash

module load python/3.7.3

# Limit numpy to a single thread
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1

# Set Parameters 
# Set Folders
KSPACEDIR="./"
RECDIR="./"

# Set Running Parameters
REALIZATIONS=100

# GENERATE NOISY AND UNDERSAMPLED K-SPACE MEASUREMENTS
for SAMPTYPE in "vardengauss" "bernoulli"
do
    for PVAL in 0.25 0.5 0.75 0.80 0.85 0.90 0.95
    do
        for NOISEVAL in 0.01 0.05 0.1 0.3
        do

          echo 'Generating Sample with' $NOISEVAL $PVAL $SAMPTYPE
          python -m mrina.gen_samples --fromdir $KSPACEDIR \
                                     --repetitions $REALIZATIONS \
                                     --origin imgs_n1 \
                                     --dest $RECDIR \
                                     --utype $SAMPTYPE \
                                     --urate $PVAL \
                                     --noisepercent $NOISEVAL
        done
    done                                
done
