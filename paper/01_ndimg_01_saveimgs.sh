# CS with respect to the true image
python3 ../saveimgs.py --numsamples 100 \
                    --maindir ./02_ndimg/ \
                    --recdir ./02_ndimg/cs/ \
                    --maskdir ./01_patterns/ \
                    --outputdir ./07_out/01_out_ndimg/img/ \
                    --savetrue \
                    --saverec \
                    --savenoise \
                    --singlechannel \
                    --imgprefix CS \
                    --usetrueasref \
                    --printlevel 1

# CS with respect to the avg image
python3 ../saveimgs.py --numsamples 100 \
                    --maindir ./02_ndimg/ \
                    --recdir ./02_ndimg/cs/ \
                    --maskdir ./01_patterns/ \
                    --outputdir ./07_out/01_out_ndimg/img/ \
                    --savetrue \
                    --saverec \
                    --savenoise \
                    --singlechannel \
                    --imgprefix CS \
                    --printlevel 1

# CSDEB with respect to the true image
python3 ../saveimgs.py --numsamples 100 \
                    --maindir ./02_ndimg/ \
                    --recdir ./02_ndimg/csdebias/ \
                    --maskdir ./01_patterns/ \
                    --outputdir ./07_out/01_out_ndimg/img/ \
                    --savetrue \
                    --saverec \
                    --savenoise \
                    --singlechannel \
                    --imgprefix CSDEB \
                    --usetrueasref \
                    --printlevel 1

# CSDEB with respect to the avg image
python3 ../saveimgs.py --numsamples 100 \
                    --maindir ./02_ndimg/ \
                    --recdir ./02_ndimg/csdebias/ \
                    --maskdir ./01_patterns/ \
                    --outputdir ./07_out/01_out_ndimg/img/ \
                    --savetrue \
                    --saverec \
                    --savenoise \
                    --singlechannel \
                    --imgprefix CSDEB \
                    --printlevel 1

# OMP with respect to the true image
python3 ../saveimgs.py --numsamples 100 \
                    --maindir ./02_ndimg/ \
                    --recdir ./02_ndimg/omp/ \
                    --maskdir ./01_patterns/ \
                    --outputdir ./07_out/01_out_ndimg/img/ \
                    --savetrue \
                    --saverec \
                    --savenoise \
                    --singlechannel \
                    --imgprefix OMP \
                    --usetrueasref \
                    --printlevel 1

# OMP with respect to the avg image
python3 ../saveimgs.py --numsamples 100 \
                    --maindir ./02_ndimg/ \
                    --recdir ./02_ndimg/omp/ \
                    --maskdir ./01_patterns/ \
                    --outputdir ./07_out/01_out_ndimg/img/ \
                    --savetrue \
                    --saverec \
                    --savenoise \
                    --singlechannel \
                    --imgprefix OMP \
                    --printlevel 1
