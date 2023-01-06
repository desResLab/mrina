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
                    --usetrueasref \
                    --printlevel 1 \
                    --savelin \
                    --limits 0.0 184.2392788833003 -1.0776501893997192 1.0873665809631348 -1.146713376045227 1.4400959014892578 -1.2205644845962524 1.3197449445724487

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
                    --printlevel 1 \
                    --savelin \
                    --limits 0.0 184.2392788833003 -1.0776501893997192 1.0873665809631348 -1.146713376045227 1.4400959014892578 -1.2205644845962524 1.3197449445724487

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
                    --usetrueasref \
                    --printlevel 1 \
                    --savelin \
                    --limits 0.0 184.2392788833003 -1.0776501893997192 1.0873665809631348 -1.146713376045227 1.4400959014892578 -1.2205644845962524 1.3197449445724487

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
                    --printlevel 1 \
                    --savelin \
                    --limits 0.0 184.2392788833003 -1.0776501893997192 1.0873665809631348 -1.146713376045227 1.4400959014892578 -1.2205644845962524 1.3197449445724487

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
                    --usetrueasref \
                    --printlevel 1 \
                    --savelin \
                    --limits 0.0 184.2392788833003 -1.0776501893997192 1.0873665809631348 -1.146713376045227 1.4400959014892578 -1.2205644845962524 1.3197449445724487

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
                    --printlevel 1 \
                    --savelin \
                    --limits 0.0 184.2392788833003 -1.0776501893997192 1.0873665809631348 -1.146713376045227 1.4400959014892578 -1.2205644845962524 1.3197449445724487
