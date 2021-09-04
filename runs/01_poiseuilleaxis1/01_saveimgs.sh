# CS with respect to the true image
python3 ../../saveimgs.py --numsamples 100 \
                    --maindir ./ \
                    --recdir ./CS/ \
                    --maskdir ./ \
                    --outputdir ./OUT/01_imgs/ \
                    --savetrue \
                    --savemask \
                    --saverec \
                    --savenoise \
                    --usetrueasref \
                    --printlevel 1

# CS with respect to the avg image
python3 ../../saveimgs.py --numsamples 100 \
                    --maindir ./ \
                    --recdir ./CS/ \
                    --maskdir ./ \
                    --outputdir ./OUT/01_imgs/ \
                    --savetrue \
                    --savemask \
                    --saverec \
                    --savenoise \
                    --printlevel 1

# CSDEB with respect to the true image
python3 ../../saveimgs.py --numsamples 100 \
                    --maindir ./ \
                    --recdir ./CSDEB/ \
                    --maskdir ./ \
                    --outputdir ./OUT/01_imgs/ \
                    --savetrue \
                    --savemask \
                    --saverec \
                    --savenoise \
                    --usetrueasref \
                    --printlevel 1

# CSDEB with respect to the avg image
python3 ../../saveimgs.py --numsamples 100 \
                    --maindir ./ \
                    --recdir ./CSDEB/ \
                    --maskdir ./ \
                    --outputdir ./OUT/01_imgs/ \
                    --savetrue \
                    --savemask \
                    --saverec \
                    --savenoise \
                    --printlevel 1

# OMP with respect to the true image
python3 ../../saveimgs.py --numsamples 100 \
                    --maindir ./ \
                    --recdir ./OMP/ \
                    --maskdir ./ \
                    --outputdir ./OUT/01_imgs/ \
                    --savetrue \
                    --savemask \
                    --saverec \
                    --savenoise \
                    --usetrueasref \
                    --printlevel 1

# OMP with respect to the avg image
python3 ../../saveimgs.py --numsamples 100 \
                    --maindir ./ \
                    --recdir ./OMP/ \
                    --maskdir ./ \
                    --outputdir ./OUT/01_imgs/ \
                    --savetrue \
                    --savemask \
                    --saverec \
                    --savenoise \
                    --printlevel 1
