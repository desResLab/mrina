python3 ../mseplt.py --noise 0.1 0.01 0.05 0.3 \
                  --uval 0.75 0.25 0.5 \
                  --utype vardengauss bernoulli \
                  --method cs csdebias omp \
                  --numsamples 100 \
                  --numpts 50 \
                  --dir ./02_ndimg/ \
                  --outputdir ./07_out/01_out_ndimg/mse/ \
                  --maskdir ./01_patterns/ \
                  --singlechannel \
                  --usecompleximgs \
                  --addlinearrec \
                  --usetrueimg \
                  --printlevel 1

python3 ../mseplt.py --noise 0.1 0.01 0.05 0.3 \
                  --uval 0.75 0.25 0.5 \
                  --utype vardengauss bernoulli \
                  --method cs csdebias omp \
                  --numsamples 100 \
                  --numpts 50 \
                  --dir ./02_ndimg/ \
                  --outputdir ./07_out/01_out_ndimg/mse/ \
                  --maskdir ./01_patterns/ \
                  --singlechannel \
                  --usecompleximgs \
                  --addlinearrec \
                  --printlevel 1
