python3 ../corrplt.py --noise 0.1 0.01 0.05 0.3 \
                   --uval 0.75 0.25 0.5 \
                   --utype vardengauss bernoulli \
                   --method cs csdebias omp \
                   --numsamples 100 \
                   --numpts 50 \
                   --dir ./05_idealaorta/ \
                   --outputdir ./07_out/04_out_ai/corr/ \
                   --printlevel 1

