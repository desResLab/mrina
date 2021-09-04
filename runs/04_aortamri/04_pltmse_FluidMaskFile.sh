# Complex, true image, multi-channel
python3 ../../mseplt.py --noise 0.0 \
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
                        --printlevel 1

# Complex, avg image, multi-channel
python3 ../../mseplt.py --noise 0.0 \
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
                        --printlevel 1

# Vels, true image, multi-channel
python3 ../../mseplt.py --noise 0.0 \
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
                        --printlevel 1

# Vels, avg image, multi-channel
python3 ../../mseplt.py --noise 0.0 \
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
                        --printlevel 1
