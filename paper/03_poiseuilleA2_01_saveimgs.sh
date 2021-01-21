# CS with respect to the true image
python3 ../saveimgs.py --numsamples 100 \
                    --maindir ./04_poiseuilleaxis2/ \
                    --recdir ./04_poiseuilleaxis2/cs/ \
                    --maskdir ./01_patterns/ \
                    --outputdir ./07_out/03_out_p2/img/ \
                    --savetrue \
                    --savemask \
                    --saverec \
                    --savenoise \
                    --imgprefix CS \
                    --usetrueasref \
                    --printlevel 1

# CS with respect to the avg image
python3 ../saveimgs.py --numsamples 100 \
                    --maindir ./04_poiseuilleaxis2/ \
                    --recdir ./04_poiseuilleaxis2/cs/ \
                    --maskdir ./01_patterns/ \
                    --outputdir ./07_out/03_out_p2/img/ \
                    --savetrue \
                    --savemask \
                    --saverec \
                    --savenoise \
                    --imgprefix CS \
                    --printlevel 1

# CSDEB with respect to the true image
python3 ../saveimgs.py --numsamples 100 \
                    --maindir ./04_poiseuilleaxis2/ \
                    --recdir ./04_poiseuilleaxis2/csdebias/ \
                    --maskdir ./01_patterns/ \
                    --outputdir ./07_out/03_out_p2/img/ \
                    --savetrue \
                    --savemask \
                    --saverec \
                    --savenoise \
                    --imgprefix CSDEB \
                    --usetrueasref \
                    --printlevel 1

# CSDEB with respect to the avg image
python3 ../saveimgs.py --numsamples 100 \
                    --maindir ./04_poiseuilleaxis2/ \
                    --recdir ./04_poiseuilleaxis2/csdebias/ \
                    --maskdir ./01_patterns/ \
                    --outputdir ./07_out/03_out_p2/img/ \
                    --savetrue \
                    --savemask \
                    --saverec \
                    --savenoise \
                    --imgprefix CSDEB \
                    --printlevel 1

# OMP with respect to the true image
python3 ../saveimgs.py --numsamples 100 \
                    --maindir ./04_poiseuilleaxis2/ \
                    --recdir ./04_poiseuilleaxis2/omp/ \
                    --maskdir ./01_patterns/ \
                    --outputdir ./07_out/03_out_p2/img/ \
                    --savetrue \
                    --savemask \
                    --saverec \
                    --savenoise \
                    --imgprefix OMP \
                    --usetrueasref \
                    --printlevel 1

# OMP with respect to the avg image
python3 ../saveimgs.py --numsamples 100 \
                    --maindir ./04_poiseuilleaxis2/ \
                    --recdir ./04_poiseuilleaxis2/omp/ \
                    --maskdir ./01_patterns/ \
                    --outputdir ./07_out/03_out_p2/img/ \
                    --savetrue \
                    --savemask \
                    --saverec \
                    --savenoise \
                    --imgprefix OMP \
                    --printlevel 1

