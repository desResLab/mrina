# CS with respect to the true image
python3 saveimgs.py --numsamples 100 \
                    --maindir ./03_poiseuilleaxis1/ \
                    --recdir ./03_poiseuilleaxis1/cs/ \
                    --maskdir ./01_patterns/ \
                    --outputdir ./07_out/02_out_p1/img/ \
                    --savetrue \
                    --savemask \
                    --saverec \
                    --savenoise \
                    --imgprefix CS \
                    --usetrueasref \
                    --printlevel 1

# CS with respect to the avg image
python3 saveimgs.py --numsamples 100 \
                    --maindir ./03_poiseuilleaxis1/ \
                    --recdir ./03_poiseuilleaxis1/cs/ \
                    --maskdir ./01_patterns/ \
                    --outputdir ./07_out/02_out_p1/img/ \
                    --savetrue \
                    --savemask \
                    --saverec \
                    --savenoise \
                    --imgprefix CS \
                    --printlevel 1

# CSDEB with respect to the true image
python3 saveimgs.py --numsamples 100 \
                    --maindir ./03_poiseuilleaxis1/ \
                    --recdir ./03_poiseuilleaxis1/csdebias/ \
                    --maskdir ./01_patterns/ \
                    --outputdir ./07_out/02_out_p1/img/ \
                    --savetrue \
                    --savemask \
                    --saverec \
                    --savenoise \
                    --imgprefix CSDEB \
                    --usetrueasref \
                    --printlevel 1

# CSDEB with respect to the avg image
python3 saveimgs.py --numsamples 100 \
                    --maindir ./03_poiseuilleaxis1/ \
                    --recdir ./03_poiseuilleaxis1/csdebias/ \
                    --maskdir ./01_patterns/ \
                    --outputdir ./07_out/02_out_p1/img/ \
                    --savetrue \
                    --savemask \
                    --saverec \
                    --savenoise \
                    --imgprefix CSDEB \
                    --printlevel 1

# OMP with respect to the true image
python3 saveimgs.py --numsamples 100 \
                    --maindir ./03_poiseuilleaxis1/ \
                    --recdir ./03_poiseuilleaxis1/omp/ \
                    --maskdir ./01_patterns/ \
                    --outputdir ./07_out/02_out_p1/img/ \
                    --savetrue \
                    --savemask \
                    --saverec \
                    --savenoise \
                    --imgprefix OMP \
                    --usetrueasref \
                    --printlevel 1

# OMP with respect to the avg image
python3 saveimgs.py --numsamples 100 \
                    --maindir ./03_poiseuilleaxis1/ \
                    --recdir ./03_poiseuilleaxis1/omp/ \
                    --maskdir ./01_patterns/ \
                    --outputdir ./07_out/02_out_p1/img/ \
                    --savetrue \
                    --savemask \
                    --saverec \
                    --savenoise \
                    --imgprefix OMP \
                    --printlevel 1

