# Complex, true image, multi-channel
python3 mseplt.py --noise 0.0 \
                  --uval 0.75 \
                  --utype vardengauss bernoulli \
                  --method cs csdebias omp \
                  --numsamples 100 \
                  --numpts 50 \
                  --dir ./06_aortamri/ \
                  --outputdir ./07_out/05_out_amri/mse_fluid/ \
                  --maskdir ./01_patterns/ \
                  --usefluidmask \
                  --usecompleximgs \
                  --addlinearrec \
                  --usetrueimg \
                  --printlevel 1

# Complex, avg image, multi-channel
python3 mseplt.py --noise 0.0 \
                  --uval 0.75 \
                  --utype vardengauss bernoulli \
                  --method cs csdebias omp \
                  --numsamples 100 \
                  --numpts 50 \
                  --dir ./06_aortamri/ \
                  --outputdir ./07_out/05_out_amri/mse_fluid/ \
                  --maskdir ./01_patterns/ \
                  --usefluidmask \
                  --usecompleximgs \
                  --addlinearrec \
                  --printlevel 1

# Vels, true image, multi-channel
python3 mseplt.py --noise 0.0 \
                  --uval 0.75 \
                  --utype vardengauss bernoulli \
                  --method cs csdebias omp \
                  --numsamples 100 \
                  --numpts 50 \
                  --dir ./06_aortamri/ \
                  --outputdir ./07_out/05_out_amri/mse_fluid/ \
                  --maskdir ./01_patterns/ \
                  --usefluidmask \
                  --addlinearrec \
                  --usetrueimg \
                  --printlevel 1

# Vels, avg image, multi-channel
python3 mseplt.py --noise 0.0 \
                  --uval 0.75 \
                  --utype vardengauss bernoulli \
                  --method cs csdebias omp \
                  --numsamples 100 \
                  --numpts 50 \
                  --dir ./06_aortamri/ \
                  --outputdir ./07_out/05_out_amri/mse_fluid/ \
                  --maskdir ./01_patterns/ \
                  --usefluidmask \
                  --addlinearrec \
                  --printlevel 1
