Within the main directory, execute the python files. Use **genSamples.py** to generate k-space, given velocity and magnitude. Undersample and add random noise.

```
python3 genSamples.py 2 vardengauss tests/04_PoiseuilleImages/axis1/
```
This will generate 2 noise realization for each noise level (1%, 5%, 10%, 30%), and one Gaussian density undersampling pattern for each undersampling ratio (25%, 50%, 75%). 

### Recovering Images
``` 
python3 recover.py 0.01 0.25 'vardengauss' 2 1 tests/04_PoiseuilleImages/axis1/ tests/04_PoiseuilleImages/axis1/results/ tests/04_PoiseuilleImages/axis1/
```
This will recover the 1% noise case, 25% Gaussian density undersampling, with 2 realizations, using 1 process, where the noisy data is located in the first directory, the recovered images will be saved to the second directory, and the undersampling patterns are located in the third directory.

## Post Processing
Files located in tests/07_PostProcess/ are used to generate plots of results, such as correlation plots and recovered images.

### Generating correlation plots

After executing recover for all the combinations, corrplt.py can be used to generate all the correlation plots.

```
python3 tests/07_PostProcess/corrplt.py 0.01 0.25 vardengauss tests/04_PoiseuilleImages/axis1/results/
```

If recover was only executed once, use the get_vals function in tests/07_PostProcess/correlation.py to save the correlation in a numpy file.

```
imgdir = './tests/04_PoiseuilleImages/axis1/'
get_vals(0.01, 0.25, 'vardengauss', 1, 100, 50, imgdir + 'results/', imgdir, imgdir) 
``` 
This saves correlations for 1% noise, 25% Gaussian density undersampling, 1 realization, points of distance up to 100, averaged over 50 points.

### MSE Histogram
To generate a histogram of the mean squared error for all combinations of noise and undersampling pattern, execute
```
python3 tests/07_PostProcess/msehist.py ./tests/04_PoiseuilleImages/axis1/ ./tests/04_PoiseuilleImages/axis1/results/
```
Additionally, within main, options may be changed, depending on which histograms are desired.

* **use_complex**: whether to compare the MSE against the complex images, or images after velocity recovery
* **use_truth**: whether to compare the MSE against the true original, or the average recovered image

### Saving images

To save the recovered image examples, use 
```
python3 saveimgs.py ./tests/04_PoiseuilleImages/axis1/results/
```
