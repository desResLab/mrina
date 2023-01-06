# CS with respect to the true image
python -m mrina.save_imgs --numsamples 100 \
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
                    --savelin \
                    --fluidmaskfile sl_mask.npy \
                    --limits 773.930 2341.383 -33.846 36.308 -29.333 33.846 -49.377 97.729

# CS with respect to the avg image
python -m mrina.save_imgs --numsamples 100 \
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
                    --fluidmaskfile sl_mask.npy \
                    --limits 773.930 2341.383 -33.846 36.308 -29.333 33.846 -49.377 97.729

# CSDEB with respect to the true image
python -m mrina.save_imgs --numsamples 100 \
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
                    --fluidmaskfile sl_mask.npy \
                    --limits 773.930 2341.383 -33.846 36.308 -29.333 33.846 -49.377 97.729

# CSDEB with respect to the avg image
python -m mrina.save_imgs --numsamples 100 \
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
                    --fluidmaskfile sl_mask.npy \
                    --limits 773.930 2341.383 -33.846 36.308 -29.333 33.846 -49.377 97.729

# OMP with respect to the true image
python -m mrina.save_imgs --numsamples 100 \
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
                    --fluidmaskfile sl_mask.npy \
                    --limits 773.930 2341.383 -33.846 36.308 -29.333 33.846 -49.377 97.729

# OMP with respect to the avg image
python -m mrina.save_imgs --numsamples 100 \
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
                    --fluidmaskfile sl_mask.npy \
                    --limits 773.930 2341.383 -33.846 36.308 -29.333 33.846 -49.377 97.729
