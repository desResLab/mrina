# CS with respect to the true image
python3 saveimgs.py --numsamples 100 \
                    --maindir ./02_ndimg/ \
                    --recdir ./02_ndimg/cs/ \
                    --maskdir ./02_ndimg/ndimg/ \
                    --outputdir ./07_out/01_out_ndimg/img/ \
                    --savetrue \
                    --savemask \
                    --saverec \
                    --savenoise \
                    --singlechannel \
                    --imgprefix CS \
                    --usetrueasref \
                    --printlevel 1

# CS with respect to the avg image
python3 saveimgs.py --numsamples 100 \
                    --maindir ./02_ndimg/ndimg/ \
                    --recdir ./02_ndimg/ndimg/cs/ \
                    --maskdir ./02_ndimg/ndimg/ \
                    --outputdir ./07_out/01_out_ndimg/img/ \
                    --savetrue \
                    --savemask \
                    --saverec \
                    --savenoise \
                    --singlechannel \
                    --imgprefix CS \
                    --printlevel 1

# CSDEB with respect to the true image
python3 saveimgs.py --numsamples 100 \
                    --maindir ./02_ndimg/ndimg/ \
                    --recdir ./02_ndimg/ndimg/csdebias/ \
                    --maskdir ./02_ndimg/ndimg/ \
                    --outputdir ./07_out/01_out_ndimg/img/ \
                    --savetrue \
                    --savemask \
                    --saverec \
                    --savenoise \
                    --singlechannel \
                    --imgprefix CSDEB \
                    --usetrueasref \
                    --printlevel 1

# CSDEB with respect to the avg image
python3 saveimgs.py --numsamples 100 \
                    --maindir ./02_ndimg/ndimg/ \
                    --recdir ./02_ndimg/ndimg/csdebias/ \
                    --maskdir ./02_ndimg/ndimg/ \
                    --outputdir ./07_out/01_out_ndimg/img/ \
                    --savetrue \
                    --savemask \
                    --saverec \
                    --savenoise \
                    --singlechannel \
                    --imgprefix CSDEB \
                    --printlevel 1

# OMP with respect to the true image
python3 saveimgs.py --numsamples 100 \
                    --maindir ./02_ndimg/ndimg/ \
                    --recdir ./02_ndimg/ndimg/omp/ \
                    --maskdir ./02_ndimg/ndimg/ \
                    --outputdir ./07_out/01_out_ndimg/img/ \
                    --savetrue \
                    --savemask \
                    --saverec \
                    --savenoise \
                    --singlechannel \
                    --imgprefix OMP \
                    --usetrueasref \
                    --printlevel 1

# OMP with respect to the avg image
python3 saveimgs.py --numsamples 100 \
                    --maindir ./02_ndimg/ndimg/ \
                    --recdir ./02_ndimg/ndimg/omp/ \
                    --maskdir ./02_ndimg/ndimg/ \
                    --outputdir ./07_out/01_out_ndimg/img/ \
                    --savetrue \
                    --savemask \
                    --saverec \
                    --savenoise \
                    --singlechannel \
                    --imgprefix OMP \
                    --printlevel 1
