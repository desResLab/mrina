python3 ../correlation.py --numsamples 100 \
                       --numpts 50 \
                       --recdir ./02_ndimg/cs/ \
                       --ptsdir ./02_ndimg/ \
                       --vencdir ./02_ndimg/ \
                       --maindir ./02_ndimg/ \
                       --usefluidmask \
                       --singlechannel \
                       --printlevel 1

python3 ../correlation.py --numsamples 100 \
                       --numpts 50 \
                       --recdir ./02_ndimg/csdebias/ \
                       --ptsdir ./02_ndimg/ \
                       --vencdir ./02_ndimg/ \
                       --maindir ./02_ndimg/ \
                       --usefluidmask \
                       --singlechannel \
                       --printlevel 1

python3 ../correlation.py --numsamples 100 \
                       --numpts 50 \
                       --recdir ./02_ndimg/omp/ \
                       --ptsdir ./02_ndimg/ \
                       --vencdir ./02_ndimg/ \
                       --maindir ./02_ndimg/ \
                       --usefluidmask \
                       --singlechannel \
                       --printlevel 1
