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
                    --savelin \
                    --usetrueasref \
                    --printlevel 1 \
                    --limits 0.0 1.0 0.0 1.0 0.0 1.0 0.0 1.0 \
                    --fluidmaskfile p2_mask.npy

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
                    --savelin \
                    --printlevel 1 \
                    --limits 0.0 1.0 0.0 1.0 0.0 1.0 0.0 1.0 \
                    --fluidmaskfile p2_mask.npy

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
                    --savelin \
                    --usetrueasref \
                    --printlevel 1 \
                    --limits 0.0 1.0 0.0 1.0 0.0 1.0 0.0 1.0 \
                    --fluidmaskfile p2_mask.npy

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
                    --savelin \
                    --printlevel 1 \
                    --limits 0.0 1.0 0.0 1.0 0.0 1.0 0.0 1.0 \
                    --fluidmaskfile p2_mask.npy

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
                    --savelin \
                    --usetrueasref \
                    --printlevel 1 \
                    --limits 0.0 1.0 0.0 1.0 0.0 1.0 0.0 1.0 \
                    --fluidmaskfile p2_mask.npy

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
                    --savelin \
                    --printlevel 1 \
                    --limits 0.0 1.0 0.0 1.0 0.0 1.0 0.0 1.0 \
                    --fluidmaskfile p2_mask.npy