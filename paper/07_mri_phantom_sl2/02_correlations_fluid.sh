python -m mrina.correlation --numsamples 100 \
                            --numpts 50 \
                            --maxcorrpixeldist 30 \
                            --recdir ./CS/ \
                            --ptsdir ./ \
                            --vencdir ./ \
                            --maindir ./ \
                            --usefluidmask \
                            --fluidmaskfile sl_mask.npy \
                            --printlevel 1

python -m mrina.correlation --numsamples 100 \
                            --numpts 50 \
                            --maxcorrpixeldist 30 \
                            --recdir ./CSDEB/ \
                            --ptsdir ./ \
                            --vencdir ./ \
                            --maindir ./ \
                            --usefluidmask \
                            --fluidmaskfile sl_mask.npy \
                            --printlevel 1

python -m mrina.correlation --numsamples 100 \
                            --numpts 50 \
                            --maxcorrpixeldist 30 \
                            --recdir ./OMP/ \
                            --ptsdir ./ \
                            --vencdir ./ \
                            --maindir ./ \
                            --usefluidmask \
                            --fluidmaskfile sl_mask.npy \
                            --printlevel 1




