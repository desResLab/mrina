# CS with respect to the true image
python3 ../../../saveimgs.py --numsamples 2 \
                             --savetrue \
                             --savemask \
                             --saverec \
                             --savenoise \
                             --imgprefix CS \
                             --usetrueasref \
                             --printlevel 1

# CS with respect to the avg image
python3 ../../../saveimgs.py --numsamples 2 \
                             --savetrue \
                             --savemask \
                             --saverec  \
                             --savenoise \
                             --imgprefix CS \
                             --printlevel 1

