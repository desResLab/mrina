# Complex, true image, multi-channel
python -m mrina.plot_mse --noise 0.1 0.05 0.3 \
                        --uval 0.85 0.90 0.95 \
                        --utype vardengauss \
                        --method cs csdebias omp \
                        --wavelet haar db8 \
                        --numsamples 100 \
                        --numpts 50 \
                        --dir ./ \
                        --outputdir ./OUT/03_mse/ \
                        --maskdir ./ \
                        --usefluidmask \
                        --fluidmaskfile ia_mask.npy \
                        --usecompleximgs \
                        --addlinearrec \
                        --usetrueimg \
                        --printlevel 1 \
                        --percstring 1

# Complex, avg image, multi-channel
python -m mrina.plot_mse --noise 0.1 0.05 0.3 \
                        --uval 0.85 0.90 0.95 \
                        --utype vardengauss \
                        --method cs csdebias omp \
                        --wavelet haar db8 \
                        --numsamples 100 \
                        --numpts 50 \
                        --dir ./ \
                        --outputdir ./OUT/03_mse/ \
                        --maskdir ./ \
                        --usefluidmask \
                        --fluidmaskfile ia_mask.npy \
                        --usecompleximgs \
                        --addlinearrec \
                        --printlevel 1 \
                        --percstring 2

# Vels, true image, multi-channel
python -m mrina.plot_mse --noise 0.1 0.05 0.3 \
                        --uval 0.85 0.90 0.95 \
                        --utype vardengauss \
                        --method cs csdebias omp \
                        --wavelet haar db8 \
                        --numsamples 100 \
                        --numpts 50 \
                        --dir ./ \
                        --outputdir ./OUT/03_mse/ \
                        --maskdir ./ \
                        --usefluidmask \
                        --fluidmaskfile ia_mask.npy \
                        --addlinearrec \
                        --usetrueimg \
                        --printlevel 1 \
                        --percstring 3

# Vels, avg image, multi-channel
python -m mrina.plot_mse --noise 0.1 0.05 0.3 \
                        --uval 0.85 0.90 0.95 \
                        --utype vardengauss \
                        --method cs csdebias omp \
                        --wavelet haar db8 \
                        --numsamples 100 \
                        --numpts 50 \
                        --dir ./ \
                        --outputdir ./OUT/03_mse/ \
                        --maskdir ./ \
                        --usefluidmask \
                        --fluidmaskfile ia_mask.npy \
                        --addlinearrec \
                        --printlevel 1 \
                        --percstring 4
