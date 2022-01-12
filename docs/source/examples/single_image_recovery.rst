 ## Single-image examples

Example of recovering a 64x64 pixel image from its undersampled frequency information using a Gaussian mask in k-space, 75% undersampling (only 1 every 4 frequencies is retained) and adding a SNR equal to 50.

<table>
  <tr>
    <td>Original image</td>
    <td> <img src="resources/im.jpg"  alt="1" width = 200px ></td>
    <td>Wavelet transform</td>
    <td> <img src="resources/wim.jpg"  alt="1" width = 200px ></td>
  </tr> 
  <tr>
    <td>k-space mask</td>  
    <td> <img src="resources/mask.jpg"  alt="1" width = 200px ></td>
    <td>Noisy k-space measurements</td>  
    <td> <img src="resources/y.jpg"  alt="1" width = 200px ></td>
  <tr>
    <td>Noiseless reconstruction: </td>  
    <td> <img src="resources/imrec_noiseless.jpg"  alt="1" width = 200px ></td>
    <td>Reconstruction: CS</td>  
    <td> <img src="resources/imrec_noisy.jpg"  alt="1" width = 200px ></td>
  </tr>
  <tr>
    <td>Reconstruction: CSDEB</td>    
    <td> <img src="resources/imrec_noisy_bd.jpg"  alt="1" width = 200px ></td>    
    <td>Reconstruction: stOMP</td>  
    <td> <img src="resources/imrec_noisy_omp.jpg"  alt="1" width = 200px ></td>    
  </tr>
</table>

### Read grayscale image
```python
import cv2
im = cv2.imread('city.png', cv2.IMREAD_GRAYSCALE)/255.0
```
### Generate undersampling mask
```python
from mrina import generateSamplingMask

# Set an undesampling ratio (refers to the frequencies that are dropped)
delta = 0.75
# Generate an undersampling mask
omega = generateSamplingMask(im.shape, delta, 'bernoulli')
# Verify the undersampling ratio
nsamp = np.sum((omega == 1).ravel())/np.prod(omega.shape)
print('Included frequencies: %.1f%%' % (nsamp*100))
```
### Compute and show wavelet representation
```python
import pywt

waveName = 'haar'
waveMode = 'zero'
wim = pywt.coeffs_to_array(pywt.wavedec2(im, wavelet=waveName, mode=waveMode))[0]
plt.figure(figsize=(8,8))
plt.imshow(np.log(np.abs(wim)+1.0e-5), cmap='gray')
plt.axis('off')
plt.show()
```
### Initialize a WaveletToFourier operator and generate noiseless k-space measurements
```python
from mrina import OperatorWaveletToFourier

# Create a new operator
A = OperatorWaveletToFourier(im.shape, samplingSet=omega[0], waveletName=waveName)
yim = A.eval(wim, 1)
```
### Noiseless recovery using l1-norm minimization
```python
from mrina import RecoveryL1NormNoisy

# Recovery - for low values of eta it is better to use SoS-L1Ball
wimrec_cpx, _ = RecoveryL1NormNoisy(0.01, yim, A, disp=True, method='SoS-L1Ball')
# The recovered coefficients could be complex.
imrec_cpx = A.getImageFromWavelet(wimrec_cpx)
imrec = np.abs(imrec_cpx)
```
### Generate noise in the frequency domain
```python
# Target SNR
SNR = 50
# Signal power. The factor 2 accounts for real/imaginary parts
yim_pow = la.norm(yim.ravel()) ** 2 / (2 * yim.size)
# Set noise standard deviation
sigma = np.sqrt(yim_pow / SNR)
# Add noise
y = yim + sigma * (np.random.normal(size=yim.shape) + 1j * np.random.normal(size=yim.shape))
```
### Image recovery with l1-norm minimization
```python
# Set the eta parameter
eta = np.sqrt(2 * y.size) * sigma
# Run recovery with CS
wimrec_noisy_cpx, _ = RecoveryL1NormNoisy(eta, y, A, disp=True, disp_method=False, method='BPDN')
# The recovered coefficients could be complex...
imrec_noisy = np.abs(A.getImageFromWavelet(wimrec_noisy_cpx))
```
### Estimator debiasing 
```python
# Get the support from the CS solution
wim_supp = np.where(np.abs(wimrec_noisy_cpx) > 1E-4 * la.norm(wimrec_noisy_cpx.ravel(), np.inf), True, False)
# Restrict the operator
Adeb = A.colRestrict(wim_supp)
# Solve a least-squares problem
lsqr = lsQR(Adeb)  
lsqr.solve(y[Adeb.samplingSet])
wimrec_noisy_cpx_deb = np.zeros(Adeb.wavShape,dtype=np.complex)
wimrec_noisy_cpx_deb[Adeb.basisSet] = lsqr.x[:]
# The recovered coefficients could be complex...
imrec_noisy_deb = np.abs(Adeb.getImageFromWavelet(wimrec_noisy_cpx_deb))
```
### Image recovery with stOMP
```python
from mrina import lsQR,OMPRecovery
# Run stOMP recovery
wimrec_noisy_cpx, _ = OMPRecovery(A, y)
# The recovered coefficients could be complex...
imrec_noisy_cpx = A.getImageFromWavelet(wimrec_noisy_cpx)
imrec_noisy = np.abs(imrec_noisy_cpx)
```
