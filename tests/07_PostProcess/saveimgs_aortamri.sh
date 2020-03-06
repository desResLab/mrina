# CS with respect to the true image
python3 saveimgs.py --numsamples 100 \
                    --maindir /home/dschiava/Documents/02_Documents/04_Studies/03_MRI_withCarlos/09_NewImages/aortamri/ \
                    --recdir /home/dschiava/Documents/02_Documents/04_Studies/03_MRI_withCarlos/09_NewImages/aortamri/cs/ \
                    --maskdir /home/dschiava/Documents/02_Documents/04_Studies/03_MRI_withCarlos/09_NewImages/aortamri/ \
                    --outputdir /home/dschiava/Documents/02_Documents/04_Studies/03_MRI_withCarlos/09_NewImages/out_amri \
                    --savetrue \
                    --savemask \
                    --saverec \
                    --savenoise \
                    --imgprefix CS \
                    --usetrueasref \
                    --printlevel 1
# CS with respect to the avg image
python3 saveimgs.py --numsamples 100 \
                    --maindir /home/dschiava/Documents/02_Documents/04_Studies/03_MRI_withCarlos/09_NewImages/aortamri/ \
                    --recdir /home/dschiava/Documents/02_Documents/04_Studies/03_MRI_withCarlos/09_NewImages/aortamri/cs/ \
                    --maskdir /home/dschiava/Documents/02_Documents/04_Studies/03_MRI_withCarlos/09_NewImages/aortamri/ \
                    --outputdir /home/dschiava/Documents/02_Documents/04_Studies/03_MRI_withCarlos/09_NewImages/out_amri \
                    --savetrue \
                    --savemask \
                    --saverec \
                    --savenoise \
                    --imgprefix CS \
                    --printlevel 1

# CSDEB with respect to the true image
python3 saveimgs.py --numsamples 100 \
                    --maindir /home/dschiava/Documents/02_Documents/04_Studies/03_MRI_withCarlos/09_NewImages/aortamri/ \
                    --recdir /home/dschiava/Documents/02_Documents/04_Studies/03_MRI_withCarlos/09_NewImages/aortamri/csdebias/ \
                    --maskdir /home/dschiava/Documents/02_Documents/04_Studies/03_MRI_withCarlos/09_NewImages/aortamri/ \
                    --outputdir /home/dschiava/Documents/02_Documents/04_Studies/03_MRI_withCarlos/09_NewImages/out_amri \
                    --savetrue \
                    --savemask \
                    --saverec \
                    --savenoise \
                    --imgprefix CSDEB \
                    --usetrueasref \
                    --printlevel 1
# CSDEB with respect to the avg image
python3 saveimgs.py --numsamples 100 \
                    --maindir /home/dschiava/Documents/02_Documents/04_Studies/03_MRI_withCarlos/09_NewImages/aortamri/ \
                    --recdir /home/dschiava/Documents/02_Documents/04_Studies/03_MRI_withCarlos/09_NewImages/aortamri/csdebias/ \
                    --maskdir /home/dschiava/Documents/02_Documents/04_Studies/03_MRI_withCarlos/09_NewImages/aortamri/ \
                    --outputdir /home/dschiava/Documents/02_Documents/04_Studies/03_MRI_withCarlos/09_NewImages/out_amri \
                    --savetrue \
                    --savemask \
                    --saverec \
                    --savenoise \
                    --imgprefix CSDEB \
                    --printlevel 1

# OMP with respect to the true image
python3 saveimgs.py --numsamples 100 \
                    --maindir /home/dschiava/Documents/02_Documents/04_Studies/03_MRI_withCarlos/09_NewImages/aortamri/ \
                    --recdir /home/dschiava/Documents/02_Documents/04_Studies/03_MRI_withCarlos/09_NewImages/aortamri/omp/ \
                    --maskdir /home/dschiava/Documents/02_Documents/04_Studies/03_MRI_withCarlos/09_NewImages/aortamri/ \
                    --outputdir /home/dschiava/Documents/02_Documents/04_Studies/03_MRI_withCarlos/09_NewImages/out_amri \
                    --savetrue \
                    --savemask \
                    --saverec \
                    --savenoise \
                    --imgprefix OMP \
                    --usetrueasref \
                    --printlevel 1
# OMP with respect to the avg image
python3 saveimgs.py --numsamples 100 \
                    --maindir /home/dschiava/Documents/02_Documents/04_Studies/03_MRI_withCarlos/09_NewImages/aortamri/ \
                    --recdir /home/dschiava/Documents/02_Documents/04_Studies/03_MRI_withCarlos/09_NewImages/aortamri/omp/ \
                    --maskdir /home/dschiava/Documents/02_Documents/04_Studies/03_MRI_withCarlos/09_NewImages/aortamri/ \
                    --outputdir /home/dschiava/Documents/02_Documents/04_Studies/03_MRI_withCarlos/09_NewImages/out_amri \
                    --savetrue \
                    --savemask \
                    --saverec \
                    --savenoise \
                    --imgprefix OMP \
                    --printlevel 1

