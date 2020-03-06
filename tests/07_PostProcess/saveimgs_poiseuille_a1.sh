# CS with respect to the true image
python3 saveimgs.py --numsamples 100 \
                    --maindir /home/dschiava/Documents/02_Documents/04_Studies/03_MRI_withCarlos/09_NewImages/poiseuilleaxis1/ \
                    --recdir /home/dschiava/Documents/02_Documents/04_Studies/03_MRI_withCarlos/09_NewImages/poiseuilleaxis1/cs/ \
                    --maskdir /home/dschiava/Documents/02_Documents/04_Studies/03_MRI_withCarlos/09_NewImages/poiseuilleaxis1/ \
                    --outputdir /home/dschiava/Documents/02_Documents/04_Studies/03_MRI_withCarlos/09_NewImages/out_p1 \
                    --savetrue \
                    --savemask \
                    --saverec \
                    --savenoise \
                    --imgprefix CS \
                    --usetrueasref \
                    --printlevel 1
# CS with respect to the avg image
python3 saveimgs.py --numsamples 100 \
                    --maindir /home/dschiava/Documents/02_Documents/04_Studies/03_MRI_withCarlos/09_NewImages/poiseuilleaxis1/ \
                    --recdir /home/dschiava/Documents/02_Documents/04_Studies/03_MRI_withCarlos/09_NewImages/poiseuilleaxis1/cs/ \
                    --maskdir /home/dschiava/Documents/02_Documents/04_Studies/03_MRI_withCarlos/09_NewImages/poiseuilleaxis1/ \
                    --outputdir /home/dschiava/Documents/02_Documents/04_Studies/03_MRI_withCarlos/09_NewImages/out_p1 \
                    --savetrue \
                    --savemask \
                    --saverec \
                    --savenoise \
                    --imgprefix CS \
                    --printlevel 1

# CSDEB with respect to the true image
python3 saveimgs.py --numsamples 100 \
                    --maindir /home/dschiava/Documents/02_Documents/04_Studies/03_MRI_withCarlos/09_NewImages/poiseuilleaxis1/ \
                    --recdir /home/dschiava/Documents/02_Documents/04_Studies/03_MRI_withCarlos/09_NewImages/poiseuilleaxis1/csdebias/ \
                    --maskdir /home/dschiava/Documents/02_Documents/04_Studies/03_MRI_withCarlos/09_NewImages/poiseuilleaxis1/ \
                    --outputdir /home/dschiava/Documents/02_Documents/04_Studies/03_MRI_withCarlos/09_NewImages/out_p1 \
                    --savetrue \
                    --savemask \
                    --saverec \
                    --savenoise \
                    --imgprefix CSDEB \
                    --usetrueasref \
                    --printlevel 1
# CSDEB with respect to the avg image
python3 saveimgs.py --numsamples 100 \
                    --maindir /home/dschiava/Documents/02_Documents/04_Studies/03_MRI_withCarlos/09_NewImages/poiseuilleaxis1/ \
                    --recdir /home/dschiava/Documents/02_Documents/04_Studies/03_MRI_withCarlos/09_NewImages/poiseuilleaxis1/csdebias/ \
                    --maskdir /home/dschiava/Documents/02_Documents/04_Studies/03_MRI_withCarlos/09_NewImages/poiseuilleaxis1/ \
                    --outputdir /home/dschiava/Documents/02_Documents/04_Studies/03_MRI_withCarlos/09_NewImages/out_p1 \
                    --savetrue \
                    --savemask \
                    --saverec \
                    --savenoise \
                    --imgprefix CSDEB \
                    --printlevel 1

# OMP with respect to the true image
python3 saveimgs.py --numsamples 100 \
                    --maindir /home/dschiava/Documents/02_Documents/04_Studies/03_MRI_withCarlos/09_NewImages/poiseuilleaxis1/ \
                    --recdir /home/dschiava/Documents/02_Documents/04_Studies/03_MRI_withCarlos/09_NewImages/poiseuilleaxis1/omp/ \
                    --maskdir /home/dschiava/Documents/02_Documents/04_Studies/03_MRI_withCarlos/09_NewImages/poiseuilleaxis1/ \
                    --outputdir /home/dschiava/Documents/02_Documents/04_Studies/03_MRI_withCarlos/09_NewImages/out_p1 \
                    --savetrue \
                    --savemask \
                    --saverec \
                    --savenoise \
                    --imgprefix OMP \
                    --usetrueasref \
                    --printlevel 1
# OMP with respect to the avg image
python3 saveimgs.py --numsamples 100 \
                    --maindir /home/dschiava/Documents/02_Documents/04_Studies/03_MRI_withCarlos/09_NewImages/poiseuilleaxis1/ \
                    --recdir /home/dschiava/Documents/02_Documents/04_Studies/03_MRI_withCarlos/09_NewImages/poiseuilleaxis1/omp/ \
                    --maskdir /home/dschiava/Documents/02_Documents/04_Studies/03_MRI_withCarlos/09_NewImages/poiseuilleaxis1/ \
                    --outputdir /home/dschiava/Documents/02_Documents/04_Studies/03_MRI_withCarlos/09_NewImages/out_p1 \
                    --savetrue \
                    --savemask \
                    --saverec \
                    --savenoise \
                    --imgprefix OMP \
                    --printlevel 1

