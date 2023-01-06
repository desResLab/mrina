# Complex, true image, multi-channel
python -m mrina.plot_mse --noise 0.1 \
                        --uval 0.75 0.25 0.5 \
                        --utype vardengauss bernoulli \
                        --method cs csdebias omp \
                        --wavelet haar db8 \
                        --numsamples 100 \
                        --numpts 50 \
                        --dir ./ \
                        --outputdir ./OUT/03_mse/ \
                        --maskdir ./ \
                        --usecompleximgs \
                        --addlinearrec \
                        --usetrueimg \
                        --usefluidmask \
                        --fluidmaskfile mriaorta_mask.npy \
                        --printlevel 1 \
                        --percstring 1

# Complex, avg image, multi-channel
python -m mrina.plot_mse --noise 0.1 \
                        --uval 0.75 0.25 0.5 \
                        --utype vardengauss bernoulli \
                        --method cs csdebias omp \
                        --wavelet haar db8 \
                        --numsamples 100 \
                        --numpts 50 \
                        --dir ./ \
                        --outputdir ./OUT/03_mse/ \
                        --maskdir ./ \
                        --usecompleximgs \
                        --addlinearrec \
                        --usefluidmask \
                        --fluidmaskfile mriaorta_mask.npy \
                        --printlevel 1 \
                        --percstring 2

# Vels, true image, multi-channel
python -m mrina.plot_mse --noise 0.1 \
                        --uval 0.75 0.25 0.5 \
                        --utype vardengauss bernoulli \
                        --method cs csdebias omp \
                        --wavelet haar db8 \
                        --numsamples 100 \
                        --numpts 50 \
                        --dir ./ \
                        --outputdir ./OUT/03_mse/ \
                        --maskdir ./ \
                        --addlinearrec \
                        --usetrueimg \
                        --usefluidmask \
                        --fluidmaskfile mriaorta_mask.npy \
                        --printlevel 1 \
                        --percstring 3

# Vels, avg image, multi-channel
python -m mrina.plot_mse --noise 0.1 \
                        --uval 0.75 0.25 0.5 \
                        --utype vardengauss bernoulli \
                        --method cs csdebias omp \
                        --wavelet haar db8 \
                        --numsamples 100 \
                        --numpts 50 \
                        --dir ./ \
                        --outputdir ./OUT/03_mse/ \
                        --maskdir ./ \
                        --addlinearrec \
                        --usefluidmask \
                        --fluidmaskfile mriaorta_mask.npy \
                        --printlevel 1 \
                        --percstring 4
