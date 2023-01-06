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
                    --limits 538.812 1929.926 -9.113 31.385 -53.890 13.919 -13.685 22.007
                    
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
                    --limits 538.812 1929.926 -9.113 31.385 -53.890 13.919 -13.685 22.007
                    
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
                    --limits 538.812 1929.926 -9.113 31.385 -53.890 13.919 -13.685 22.007

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
                    --limits 538.812 1929.926 -9.113 31.385 -53.890 13.919 -13.685 22.007

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
                    --limits 538.812 1929.926 -9.113 31.385 -53.890 13.919 -13.685 22.007

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
                    --limits 538.812 1929.926 -9.113 31.385 -53.890 13.919 -13.685 22.007
                    