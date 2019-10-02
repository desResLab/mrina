#!/bin/bash
DIR="./axis1/"
RECDIR="./axis1/results/"
NOISE=0.1
P=0.5
PATTERN="vardengauss"
PROCESSES=2
REALIZATIONS=2
SOLVER_MODE=0 #0 cs, 1 csdebias, 2 omp

for SAMPTYPE in "vardengauss" "bernoulli"
do
    echo $SAMPTYPE
    python3 ../../genSamples.py $REALIZATIONS $SAMPTYPE $DIR
done

#python3 ../../recover.py $NOISE $P $PATTERN $REALIZATIONS $PROCESSES $DIR $RECDIR $DIR

for SAMPTYPE in "vardengauss" "bernoulli"
do
    for PVAL in 0.25 0.5 0.75
    do
        for NOISEVAL in 0.01 0.05 0.1 0.3
        do
            echo $NOISEVAL $PVAL $SAMPTYPE
            python3 ../../recover.py $NOISEVAL $PVAL $SAMPTYPE $REALIZATIONS $PROCESSES $DIR $RECDIR $DIR $SOLVER_MODE
        done
    done
done

#post processing
python3 ../07_PostProcess/corrplt.py $NOISE $P $PATTERN $REALIZATIONS $RECDIR $DIR $DIR
python3 ../07_PostProcess/mseplt.py $REALIZATIONS $DIR $RECDIR
python3 ../07_PostProcess/saveimgs.py $REALIZATIONS $DIR $RECDIR $DIR

