# reconstruction

#### CSRecoverySuite.py
Reconstruct, given Fourier space data.
Parameters for CSRecovery() method: 
* method: pgdl1, l1reg
* disp: 0 - display nothing. 1 - display summary but don't display the solver iterations. 2 - display summary, solver iterations, and main iterations.

### Generating noisy samples
Use genSamples.py to generate kspace, given velocity and magnitude. Undersample and add random noise.
Run main, changing parameters to samples() method such that:
* directory: location of vtk files
* numRealizations: number of noisy images
* uType: undersampling type. options: 'poisson','halton','bernoulli'
* sliceIndex: index of slice (0,1,2) for ex. 0 for the orthogonal slice, and 2 for the second Poiseuille slice
* npydir: location to store the generated samples

### Recovering Images
``` 
python3 recover.py 0.01 0.25 'bernoulli' 100 
```
where parameters are listed in order: 
* noise level: ex. 0.01 means 1% noise
* undersampled percentage: ex. 0.25 means 25% of the image not sampled
* undersampling type
* number of realizations: should match file name generated in genSamples.py

If running on CRC, be sure to include:
``` setenv OMP_NUM_THREADS 1 ```

Within main of recover.py, specify parameter to recoverAll() method c = number of processors to use. If images have already been recovered, comment out lines 127-128, and uncomment lines 129 ```recovered = np.load(...)``` so that the recovery doesn't have to be executed again. 

#### circle/CSRecoverySuite.py
As shown in circle/demo_csrecovery_circle.py, to include debiasing, execute the following commands, where ```deb_csim ``` is the final recovered image after debiasing.
```
#   Recovery via CS
cswim, fcwim    = CSRecovery(eta, yim, A, np.zeros( wsz ), disp=3);
csim            = pywt.waverec2(array2pywt(cswim), wavelet=wvlt, mode='periodic');
#   Debiasing CS reconstruction
deb_cswim, deb_fcwim  = CSRecoveryDebiasing(yim, A, cswim)
deb_csim        = pywt.waverec2(array2pywt(deb_cswim), wavelet=wvlt, mode='periodic');
```

### Generating correlation plots
In test/02_TestImage/test_correlation.py, execute the method ```get_all()``` to get the correlation for all tests mentioned in paper. To get the correlations for a specific noise percent and undersampling percent, execute
``` 
get_vals(noise_percent, undersampling_percent, num_realizations, size, num_pts) 
``` 
where
* size: the maximum distance to compute correlations. Points are generated starting from distance 1 to distance ```size```
* num_pts: the number of points to average the correlation across. If num_pts = 20, then 20 pairs of points are selected for each distance d.

After executing ```get_all```, you can run test/02_TestImage/jointplt.py to generate the comparison plots.
