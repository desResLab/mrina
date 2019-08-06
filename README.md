## Compressed Sensing Image Recovery Suite

### CSRecoverySuite.py

A set of tools is provided to perform image reconstruct from undersampled and noisy k-space data.

The steps below summarize how to use the tool. We first create synthetic undersampled and noisy Fourier measurements from a complex image, and then reconstruct it using the tools in the package. 

#### Generating synthetic Fourier k-space observations

We start by reading an grayscale image using openCV.
```
import cv2
im = cv2.imread(fileName, cv2.IMREAD_GRAYSCALE)
im = crop(im)
```
The *crop()* function crops the image to power of 2 dimensions, so the wavelet transform provided by the *pywt* is orthogonal.
We then generate a Gaussian undersampling mask, using the *generateSamplingMask()* function, where *delta* is the undersampling ratio
```
from CSRecoverySuite import generateSamplingMask
omega = generateSamplingMask(im.shape, delta, 'gaussian')
nsamp = np.sum(np.where( omega, 1, 0 ).ravel())
```
Possible choices are ```'bernoulli'```, ```'triangle'```, ```'gaussian'```, ```'exponential'```.
We set the wavelet padding as
```
waveMode = 'periodization'
```
and compute the 2D Haar wavelet transform using
```
wim = pywt2array(pywt.wavedec2(im, wavelet='haar', mode=waveMode), im.shape)
```
#### Definition of the measurement operator
Non linear reconstruction is achieved with the solution of the following problem

```
A = Operator4dFlow(imsz=imsz, insz=wim.shape, samplingSet=omega, waveletName='haar', waveletMode=waveMode)
```


```
A.input_size():
```

```
A.shape():
```

Once the operator is defined we can apply to any wavelet image using the overloaded ```*``` multiplication. 
```
c = A*b
```

```
b = A.T*c
```

```
A.getNorm()
```

```
B = A.colRestrict(idxSet)
```

#### Add noise to the Fourier data
```
yim = A*wim
```
We then define the noise level as
```
# Noisy data
nlevel      = nlevelVal
imNrm       = np.linalg.norm(im.ravel(), ord=2)
sigma       = nlevel * (imNrm / np.sqrt(2 * im.size))
y           = yim + sigma * (np.random.normal(size=yim.shape) + 1j * np.random.normal(size=yim.shape))
```

#### Reconstruct original image from undersampled noisy Fourier measurements *yim*

```
# Recovery via orthogonal projection
#   FOURIER TRANSFORM NEEDS TO BE NORMALIZED
fim             = (im.size ** -1/2) * fft.fft2( im )
fim[~omega]     = 0
fim             = (im.size ** -1/2) * fft.ifft2( fim )
```



```
eta             = sigma * (np.sqrt(2 * nsamp) + 1.6)
cswim, fcwim = CSRecovery(eta, yim, A, np.zeros( wsz ), disp=3)
csim = pywt.waverec2(array2pywt(cswim), wavelet='haar', mode=waveMode)
```


```
deb_cswim, deb_fcwim = CSRecoveryDebiasing(yim, A, cswim)
deb_csim = pywt.waverec2(array2pywt(deb_cswim), wavelet='haar', mode=waveMode)
```

```
omp_cswim, omp_fcwim = OMPRecovery(A, yim, tol=5.0e-2)
omp_csim = pywt.waverec2(array2pywt(omp_cswim), wavelet='haar', mode=waveMode)
```

### Generating noisy samples

Use **genSamples.py** to generate k-space, given velocity and magnitude. Undersample and add random noise.

Run main, changing parameters to samples() method such that:

* **directory**: location of VTK files
* **numRealizations**: number of noisy images
* **uType**: undersampling type. options: 'poisson','halton','bernoulli'
* **sliceIndex**: index of slice (0,1,2) for ex. 0 for the orthogonal slice, and 2 for the second Poiseuille slice
* **npydir**: location to store the generated samples

### Recovering Images
``` 
python3 recover.py 0.01 0.25 'bernoulli' 100 
```
where parameters are listed in order: 
* noise level: ex. 0.01 means 1% noise
* undersampled percentage: ex. 0.25 means 25% of the image not sampled
* undersampling type
* number of realizations: should match file name generated in genSamples.py

If running on CRC, be sure to include

``` setenv OMP_NUM_THREADS 1 ```

so the numerical libraries invoked by numpy use only 1 thread per code, not overloading the processors. 

Within main of recover.py, specify parameter to recoverAll() method c = number of processors to use. If images have already been recovered, comment out lines 127-128, and uncomment lines 129 ```recovered = np.load(...)``` so that the recovery doesn't have to be executed again. 

### Generating correlation plots

In test/02_TestImage/test_correlation.py, execute the method ```get_all()``` to get the correlation for all tests mentioned in paper. To get the correlations for a specific noise percent and undersampling percent, execute

``` 
get_vals(noise_percent, undersampling_percent, num_realizations, size, num_pts) 
``` 

where:

* **size**: the maximum distance to compute correlations. Points are generated starting from distance 1 to distance ```size```
* **num_pts**: the number of points to average the correlation across. If num_pts = 20, then 20 pairs of points are selected for each distance d.

After executing ```get_all```, you can run test/02_TestImage/jointplt.py to generate the comparison plots.