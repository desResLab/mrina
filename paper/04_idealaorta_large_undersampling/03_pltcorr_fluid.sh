python -m mrina.plot_corr --noise 0.1 0.05 0.3 \
                         --uval 0.85 0.90 0.95 \
                         --utype vardengauss \
                         --method cs csdebias omp \
                         --wavelet haar db8 \
                         --numsamples 100 \
                         --numpts 50 \
                         --dir ./ \
                         --outputdir ./OUT/02_corr/ \
                         --usefluidmask \
                         --printlevel 1

