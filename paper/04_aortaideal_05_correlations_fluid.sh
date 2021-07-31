python3 ../correlation.py --numsamples 100 \
                       --numpts 50 \
                       --recdir ./05_idealaorta/cs/ \
                       --ptsdir ./05_idealaorta/ \
                       --vencdir ./05_idealaorta/ \
                       --maindir ./05_idealaorta/ \
                       --usefluidmask \
                       --printlevel 1

python3 ../correlation.py --numsamples 100 \
                       --numpts 50 \
                       --recdir ./05_idealaorta/csdebias/ \
                       --ptsdir ./05_idealaorta/ \
                       --vencdir ./05_idealaorta/ \
                       --maindir ./05_idealaorta/ \
                       --usefluidmask \
                       --printlevel 1

python3 ../correlation.py --numsamples 100 \
                       --numpts 50 \
                       --recdir ./05_idealaorta/omp/ \
                       --ptsdir ./05_idealaorta/ \
                       --vencdir ./05_idealaorta/ \
                       --maindir ./05_idealaorta/ \
                       --usefluidmask \
                       --printlevel 1




