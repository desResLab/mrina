# Complex, true image, multi-channel
# --usecompleximgs \
python -m mrina.plot_mse --noise 0.1 0.01 0.05 0.3 \
                        --uval 0.75 0.25 0.5 \
                        --utype vardengauss bernoulli \
                        --method cs csdebias omp \
                        --wavelet haar db8 \
                        --numsamples 100 \
                        --numpts 50 \
                        --dir ./ \
                        --outputdir ./OUT/03_mse/ \
                        --maskdir ./ \
                        --usefluidmask \
                        --usetrueimg \
                        --addlinearrec \
                        --printlevel 1
