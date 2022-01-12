## Script functionalities 

MRIna also provides scripts to automate:

- the generation of noisy k-space signals.
- linear and non-linear image reconstruction.
- post-processing of reconstructed images.

### Image data

The image data should be stored on a numpy tensor in *npy* format with shape (r, i, n, im_1, im_2), where:

+ r is the number of image repetitions.
+ i is the image number. For 4D flow MRI you need 4 images, i.e., one density and three velocity components. 
+ n has a single index.
+ im_1,im_2 are the two image dimensions.

The image file name is typically *imgs_n1.npy*.

### Common parameters

For the example below, the following parameters are specified

```sh
FOLDER="./" # Folder name, here a single folder is used for all tasks
REALIZATIONS=3 # Number of realizations
IMGNAME="imgs_n1" # Name is the original image
SAMPTYPE="vardengauss" # Undersampling mask pattern
UVAL=0.75 # Undersampling ratio (75% frequencies dropped) 
NOISEVAL=0.1 # Noise internsity as a fraction of the average k-space signal norm
PROCESSES=1 # Number of parallel processes for reconstruction (shared memory only)
SOLVERMODE=2 # Recovery algorithm (0-CS, 1-CSDEB, 2-OMP)
METHOD="omp"  # Recovery algorithm (cs, csdeb or omp)
WAVETYPE="haar" # Selected wavelet frame
PRINTLEV=1 # Print level (the larger the more verbose)
NUMPOINTS=10 # Number of poits pairs for computing currelation
```
### Sample generation
```sh
python -m mrina.gen_samples --fromdir $FOLDER \
                            --repetitions $REALIZATIONS \
                            --origin $IMGNAME \
                            --dest $FOLDER \
                            --utype $SAMPTYPE \
                            --urate $UVAL \
                            --noisepercent $NOISEVAL
```
For additional information on the script input parameters, type
```
python -m mrina.gen_samples --help
```
### Image recovery from noisy and undersampled k-space signal
```sh
python -m mrina.recover --noisepercent $NOISEVAL \
                        --urate $UVAL \
                        --utype $SAMPTYPE \
                        --repetitions $REALIZATIONS \
                        --numprocesses $PROCESSES \
                        --fromdir $FOLDER \
                        --recdir $FOLDER \
                        --maskdir $FOLDER \
                        --method $SOLVERMODE \
                        --wavelet $WAVETYPE \
                        --savevels
```
For additional information on the script input parameters, type
```
python -m mrina.recover --help
```
### Post-processing - Saving reconstructed images
```sh
python -m mrina.save_imgs --numsamples $REALIZATIONS \
                          --maindir $FOLDER \
                          --recdir $FOLDER \
                          --maskdir $FOLDER \
                          --outputdir $FOLDER \
                          --savetrue \
                          --savemask \
                          --saverec \
                          --savenoise \
                          --savelin \
                          --usetrueasref \
                          --printlevel $PRINTLEV \
                          --savelin
```
For additional information on the script input parameters, type
```
python -m mrina.saveimgs --help
```
### Post-processing - Computing correlations
```sh
python -m mrina.correlation --numsamples $REALIZATIONS \
                            --numpts $NUMPOINTS \
                            --maxcorrpixeldist 10 \
                            --recdir $FOLDER \
                            --ptsdir $FOLDER \
                            --vencdir $FOLDER \
                            --maindir $FOLDER \
                            --printlevel 1
```
For additional information on the script input parameters, type
```
python -m mrina.correlation --help
```
### Post-processing - Plot correlations
```sh
python -m mrina.plot_corr --noise $NOISEVAL \
                          --uval $UVAL \
                          --utype $SAMPTYPE \
                          --method $METHOD \
                          --wavelet $WAVETYPE \
                          --numsamples $REALIZATIONS \
                          --numpts $NUMPOINTS \
                          --dir $FOLDER \
                          --outputdir $FOLDER \
                          --printlevel 1

```
For additional information on the script input parameters, type
```
python -m mrina.plot_corr --help
```
### Post-processing - Compute MSE and relative errors
```sh
python -m mrina.plot_mse --noise $NOISEVAL \
                         --uval $UVAL \
                         --utype $SAMPTYPE \
                         --method $METHOD \
                         --wavelet $WAVETYPE \
                         --numsamples $REALIZATIONS \
                         --numpts $NUMPOINTS \
                         --dir $FOLDER \
                         --outputdir $FOLDER \
                         --maskdir $FOLDER \
                         --usecompleximgs \
                         --addlinearrec \
                         --usetrueimg \
                         --printlevel 1
```
For additional information on the script input parameters, type
```
python -m mrina.plot_mse --help
```
