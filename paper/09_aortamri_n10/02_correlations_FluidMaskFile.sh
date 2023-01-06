python -m mrina.correlation --numsamples 100 \
                             --numpts 50 \
                             --recdir ./CS/ \
                             --ptsdir ./ \
                             --vencdir ./ \
                             --maindir ./ \
                             --usefluidmask \
                             --fluidmaskfile mriaorta_mask.npy \
                             --printlevel 1

python -m mrina.correlation --numsamples 100 \
                             --numpts 50 \
                             --recdir ./CSDEB/ \
                             --ptsdir ./ \
                             --vencdir ./ \
                             --maindir ./ \
                             --usefluidmask \
                             --fluidmaskfile mriaorta_mask.npy \
                             --printlevel 1

python -m mrina.correlation --numsamples 100 \
                             --numpts 50 \
                             --recdir ./OMP/ \
                             --ptsdir ./ \
                             --vencdir ./ \
                             --maindir ./ \
                             --usefluidmask \
                             --fluidmaskfile mriaorta_mask.npy \
                             --printlevel 1




