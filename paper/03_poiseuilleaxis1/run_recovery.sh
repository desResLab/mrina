#!/bin/bash
KSPACEDIR="./"
RECDIR="./"
PATTERNDIR="./"
WAVELETTYPE="db4"
PROCESSES=2
REALIZATIONS=2
SOLVERMODE=2 #0 cs, 1 csdebias, 2 omp

# GENERATE NOISY AND UNDERSAMPLED K-SPACE MEASUREMENTS

for SAMPTYPE in "vardengauss" #"bernoulli"
do
    for PVAL in 0.25 #0.5 0.75
    do
        for NOISEVAL in 0.01 #0.05 0.1 0.3
        do

          echo 'Generating Sample with' $NOISEVAL $PVAL $SAMPTYPE
          python3 ../../genSamples.py --fromdir $KSPACEDIR \
                                      --repetitions $REALIZATIONS \
                                      --origin imgs_n1 \
                                      --dest $RECDIR \
                                      --utype $SAMPTYPE \
                                      --urate $PVAL \
                                      --noisepercent $NOISEVAL
        done
    done                                
done

# RECONSTRUCT IMAGES

for SAMPTYPE in "vardengauss" #"bernoulli"
do
    for PVAL in 0.25 #0.50 0.75
    do
        for NOISEVAL in 0.01 #0.05 0.1 0.3
        do
            echo 'Reconstructing' $NOISEVAL $PVAL $SAMPTYPE
            python3 ../../recover.py --noisepercent $NOISEVAL \
                                     --urate $PVAL \
                                     --utype $SAMPTYPE \
                                     --repetitions $REALIZATIONS \
                                     --numprocesses $PROCESSES \
                                     --fromdir $KSPACEDIR \
                                     --recdir $RECDIR \
                                     --maskdir $PATTERNDIR \
                                     --method $SOLVERMODE \
                                     --wavelet $WAVELETTYPE \
                                     --savevels
        done
    done
done
