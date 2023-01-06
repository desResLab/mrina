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
                    --limits 803.320 1939.723 -15.678 11.223 -51.135 1.260 -11.282 26.168

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
                    --limits 803.320 1939.723 -15.678 11.223 -51.135 1.260 -11.282 26.168

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
                    --limits 803.320 1939.723 -15.678 11.223 -51.135 1.260 -11.282 26.168

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
                    --limits 803.320 1939.723 -15.678 11.223 -51.135 1.260 -11.282 26.168

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
                    --limits 803.320 1939.723 -15.678 11.223 -51.135 1.260 -11.282 26.168

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
                    --limits 803.320 1939.723 -15.678 11.223 -51.135 1.260 -11.282 26.168
