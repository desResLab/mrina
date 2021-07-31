# Complex, true image, multi-channel
python3 ../mseplt.py --noise 0.1 0.01 0.05 0.3 \
                  --uval 0.75 0.25 0.5 \
                  --utype vardengauss bernoulli \
                  --method cs csdebias omp \
                  --numsamples 100 \
                  --numpts 50 \
                  --dir ./05_idealaorta/ \
                  --outputdir ./07_out/04_out_ai/mse/ \
                  --maskdir ./01_patterns/ \
                  --usecompleximgs \
                  --addlinearrec \
                  --usetrueimg \
                  --printlevel 1

# Complex, avg image, multi-channel
python3 ../mseplt.py --noise 0.1 0.01 0.05 0.3 \
                  --uval 0.75 0.25 0.5 \
                  --utype vardengauss bernoulli \
                  --method cs csdebias omp \
                  --numsamples 100 \
                  --numpts 50 \
                  --dir ./05_idealaorta/ \
                  --outputdir ./07_out/04_out_ai/mse/ \
                  --maskdir ./01_patterns/ \
                  --usecompleximgs \
                  --addlinearrec \
                  --printlevel 1

# Vels, true image, multi-channel
python3 ../mseplt.py --noise 0.1 0.01 0.05 0.3 \
                  --uval 0.75 0.25 0.5 \
                  --utype vardengauss bernoulli \
                  --method cs csdebias omp \
                  --numsamples 100 \
                  --numpts 50 \
                  --dir ./05_idealaorta/ \
                  --outputdir ./07_out/04_out_ai/mse/ \
                  --maskdir ./01_patterns/ \
                  --addlinearrec \
                  --usetrueimg \
                  --printlevel 1

# Vels, avg image, multi-channel
python3 ../mseplt.py --noise 0.1 0.01 0.05 0.3 \
                  --uval 0.75 0.25 0.5 \
                  --utype vardengauss bernoulli \
                  --method cs csdebias omp \
                  --numsamples 100 \
                  --numpts 50 \
                  --dir ./05_idealaorta/ \
                  --outputdir ./07_out/04_out_ai/mse/ \
                  --maskdir ./01_patterns/ \
                  --addlinearrec \
                  --printlevel 1
