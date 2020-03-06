# CS with respect to the true image
python3 saveimgs.py --numsamples 100 \
                    --maindir /home/dschiava/Documents/02_Documents/04_Studies/03_MRI_withCarlos/09_NewImages/ndimg/ \
                    --recdir /home/dschiava/Documents/02_Documents/04_Studies/03_MRI_withCarlos/09_NewImages/ndimg/cs/ \
                    --maskdir /home/dschiava/Documents/02_Documents/04_Studies/03_MRI_withCarlos/09_NewImages/ndimg/ \
                    --outputdir /home/dschiava/Documents/02_Documents/04_Studies/03_MRI_withCarlos/09_NewImages/out_ndimg \
                    --savetrue \
                    --savemask \
                    --saverec \
                    --savenoise \
                    --singlechannel \
                    --imgprefix CS \
                    --usetrueasref \
                    --printlevel 1
# CS with respect to the avg image
python3 saveimgs.py --numsamples 100 \
                    --maindir /home/dschiava/Documents/02_Documents/04_Studies/03_MRI_withCarlos/09_NewImages/ndimg/ \
                    --recdir /home/dschiava/Documents/02_Documents/04_Studies/03_MRI_withCarlos/09_NewImages/ndimg/cs/ \
                    --maskdir /home/dschiava/Documents/02_Documents/04_Studies/03_MRI_withCarlos/09_NewImages/ndimg/ \
                    --outputdir /home/dschiava/Documents/02_Documents/04_Studies/03_MRI_withCarlos/09_NewImages/out_ndimg \
                    --savetrue \
                    --savemask \
                    --saverec \
                    --savenoise \
                    --singlechannel \
                    --imgprefix CS \
                    --printlevel 1

# CSDEB with respect to the true image
python3 saveimgs.py --numsamples 100 \
                    --maindir /home/dschiava/Documents/02_Documents/04_Studies/03_MRI_withCarlos/09_NewImages/ndimg/ \
                    --recdir /home/dschiava/Documents/02_Documents/04_Studies/03_MRI_withCarlos/09_NewImages/ndimg/csdebias/ \
                    --maskdir /home/dschiava/Documents/02_Documents/04_Studies/03_MRI_withCarlos/09_NewImages/ndimg/ \
                    --outputdir /home/dschiava/Documents/02_Documents/04_Studies/03_MRI_withCarlos/09_NewImages/out_ndimg \
                    --savetrue \
                    --savemask \
                    --saverec \
                    --savenoise \
                    --singlechannel \
                    --imgprefix CSDEB \
                    --usetrueasref \
                    --printlevel 1
# CSDEB with respect to the avg image
python3 saveimgs.py --numsamples 100 \
                    --maindir /home/dschiava/Documents/02_Documents/04_Studies/03_MRI_withCarlos/09_NewImages/ndimg/ \
                    --recdir /home/dschiava/Documents/02_Documents/04_Studies/03_MRI_withCarlos/09_NewImages/ndimg/csdebias/ \
                    --maskdir /home/dschiava/Documents/02_Documents/04_Studies/03_MRI_withCarlos/09_NewImages/ndimg/ \
                    --outputdir /home/dschiava/Documents/02_Documents/04_Studies/03_MRI_withCarlos/09_NewImages/out_ndimg \
                    --savetrue \
                    --savemask \
                    --saverec \
                    --savenoise \
                    --singlechannel \
                    --imgprefix CSDEB \
                    --printlevel 1

# OMP with respect to the true image
python3 saveimgs.py --numsamples 100 \
                    --maindir /home/dschiava/Documents/02_Documents/04_Studies/03_MRI_withCarlos/09_NewImages/ndimg/ \
                    --recdir /home/dschiava/Documents/02_Documents/04_Studies/03_MRI_withCarlos/09_NewImages/ndimg/omp/ \
                    --maskdir /home/dschiava/Documents/02_Documents/04_Studies/03_MRI_withCarlos/09_NewImages/ndimg/ \
                    --outputdir /home/dschiava/Documents/02_Documents/04_Studies/03_MRI_withCarlos/09_NewImages/out_ndimg \
                    --savetrue \
                    --savemask \
                    --saverec \
                    --savenoise \
                    --singlechannel \
                    --imgprefix OMP \
                    --usetrueasref \
                    --printlevel 1
# OMP with respect to the avg image
python3 saveimgs.py --numsamples 100 \
                    --maindir /home/dschiava/Documents/02_Documents/04_Studies/03_MRI_withCarlos/09_NewImages/ndimg/ \
                    --recdir /home/dschiava/Documents/02_Documents/04_Studies/03_MRI_withCarlos/09_NewImages/ndimg/omp/ \
                    --maskdir /home/dschiava/Documents/02_Documents/04_Studies/03_MRI_withCarlos/09_NewImages/ndimg/ \
                    --outputdir /home/dschiava/Documents/02_Documents/04_Studies/03_MRI_withCarlos/09_NewImages/out_ndimg \
                    --savetrue \
                    --savemask \
                    --saverec \
                    --savenoise \
                    --singlechannel \
                    --imgprefix OMP \
                    --printlevel 1

