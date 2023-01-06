python /home/dschiava/Documents/01_Development/01_PythonApps/35_new_mrina/mrina/plot_corr.py --noise 0.1 0.01 0.05 0.3 \
                         --uval 0.75 0.25 0.5 \
                         --utype vardengauss \
                         --method cs csdebias omp \
                         --wavelet haar db8 \
                         --numsamples 100 \
                         --numpts 50 \
                         --dir ./ \
                         --outputdir ./OUT/02_corr/ \
                         --usefluidmask \
                         --printlevel 1

# python -m mrina.plot_corr --noise 0.1 0.01 0.05 0.3 \
#                          --uval 0.75 0.25 0.5 \
#                          --utype vardengauss \
#                          --method cs csdebias omp \
#                          --wavelet haar db8 \
#                          --numsamples 100 \
#                          --numpts 50 \
#                          --dir ./ \
#                          --outputdir ./OUT/02_corr/ \
#                          --usefluidmask \
#                          --printlevel 1

