#!/bin/bash

#$ -M dschiavazzi@nd.edu
#$ -m abe
#$ -pe smp 24
#$ -q long
#$ -N mri_rec

module load python/3.7.3
source /afs/crc.nd.edu/user/d/dschiava/Private/16_newex_mri/mrina/bin/activate

# Limit numpy to a single thread
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1

# Set Parameters 
# Set Folders
KSPACEDIR="../"
RECDIR="./"
PATTERNDIR="../"

# Set Running Parameters
PROCESSES=24
REALIZATIONS=100

# Set Solver To Be Used For Reconstruction
SOLVERMODE=0 #CS
SAMPTYPE="vardengauss"
NOISEVAL=0.01 #0.05 0.1 0.3
PVAL=0.25 #0.50 0.75
ETAFACTOR=0.1

# RECONSTRUCT IMAGES
for WAVETYPE in "haar" "db8"
do
    echo 'Reconstructing' $NOISEVAL $PVAL $SAMPTYPE
    python -m mrina.recover --noisepercent $NOISEVAL \
                               --urate $PVAL \
                               --utype $SAMPTYPE \
                               --repetitions $REALIZATIONS \
                               --numprocesses $PROCESSES \
                               --fromdir $KSPACEDIR \
                               --recdir $RECDIR \
                               --maskdir $PATTERNDIR \
                               --method $SOLVERMODE \
                               --wavelet $WAVETYPE \
                               --etafactor $ETAFACTOR
done

