python3 ../../correlation.py --numsamples 100 \
                             --numpts 50 \
                             --recdir ./CS/ \
                             --ptsdir ./ \
                             --vencdir ./ \
                             --maindir ./ \
                             --usefluidmask \
                             --fluidmaskfile mriaorta_mask.npy \
                             --printlevel 1

python3 ../../correlation.py --numsamples 100 \
                             --numpts 50 \
                             --recdir ./CSDEB/ \
                             --ptsdir ./ \
                             --vencdir ./ \
                             --maindir ./ \
                             --usefluidmask \
                             --fluidmaskfile mriaorta_mask.npy \
                             --printlevel 1

python3 ../../correlation.py --numsamples 100 \
                             --numpts 50 \
                             --recdir ./OMP/ \
                             --ptsdir ./ \
                             --vencdir ./ \
                             --maindir ./ \
                             --usefluidmask \
                             --fluidmaskfile mriaorta_mask.npy \
                             --printlevel 1




