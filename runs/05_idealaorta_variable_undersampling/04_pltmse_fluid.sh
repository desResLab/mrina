# Complex, true image, multi-channel
python3 ../../mseplt.py --noise 0.1 0.01 0.05 0.3 \
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
                        --fluidmaskfile ia_mask.npy \
                        --usecompleximgs \
                        --addlinearrec \
                        --usetrueimg \
                        --usemultipatterns \
                        --printlevel 1 \
                        --percstring 1

# Complex, avg image, multi-channel
python3 ../../mseplt.py --noise 0.1 0.01 0.05 0.3 \
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
                        --fluidmaskfile ia_mask.npy \
                        --usecompleximgs \
                        --addlinearrec \
                        --usemultipatterns \
                        --printlevel 1 \
                        --percstring 2

# Vels, true image, multi-channel
python3 ../../mseplt.py --noise 0.1 0.01 0.05 0.3 \
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
                        --fluidmaskfile ia_mask.npy \
                        --addlinearrec \
                        --usetrueimg \
                        --usemultipatterns \
                        --printlevel 1 \
                        --percstring 3

# Vels, avg image, multi-channel
python3 ../../mseplt.py --noise 0.1 0.01 0.05 0.3 \
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
                        --fluidmaskfile ia_mask.npy \
                        --addlinearrec \
                        --usemultipatterns \
                        --printlevel 1 \
                        --percstring 4
