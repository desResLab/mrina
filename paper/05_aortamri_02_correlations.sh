python3 ../correlation.py --numsamples 100 \
                       --numpts 50 \
                       --recdir ./06_aortamri/cs/ \
                       --ptsdir ./06_aortamri/ \
                       --vencdir ./06_aortamri/ \
                       --printlevel 1

python3 ../correlation.py --numsamples 100 \
                       --numpts 50 \
                       --recdir ./06_aortamri/csdebias/ \
                       --ptsdir ./06_aortamri/ \
                       --vencdir ./06_aortamri/ \
                       --printlevel 1

python3 ../correlation.py --numsamples 100 \
                       --numpts 50 \
                       --recdir ./06_aortamri/omp/ \
                       --ptsdir ./06_aortamri/ \
                       --vencdir ./06_aortamri/ \
                       --printlevel 1


