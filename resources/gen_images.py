#!/usr/bin/env python
# coding: utf-8

# # Step-by-step verification

# In[ ]:


import sys
import numpy as np
import numpy.linalg as la
import scipy
sys.path.append('../')


# ### Load a grayscale image
# 
# We start by reading a grayscale image using `openCV`.

# In[ ]:


import cv2
import matplotlib.pyplot as plt
im = cv2.imread('tests/city.png', cv2.IMREAD_GRAYSCALE)/255.0
plt.imshow(im, cmap='gray')
# plt.colorbar()
plt.axis('off')
print('Image size: ', im.shape)
# plt.show()
plt.savefig('im.jpg',dpi=150)


# ### Generating undersampling mask in *k*-space
# 
# We generate a Bernouilli undersampling mask using the `generateSamplingMask()` function with an undersampling ratio `delta`.

# In[ ]:


from mrina import generateSamplingMask

# Select an undesampling ratio
delta = 0.75
# Generate an undersampling mask
omega = generateSamplingMask(im.shape, delta, 'vardengauss')
# Verify the undersampling ratio
nsamp = np.sum((omega == 1).ravel())/np.prod(omega.shape)
print('Included frequencies: %.1f%%' % (nsamp*100))
# Plot mask
plt.imshow(omega[0], cmap='binary')
plt.axis('off')
# plt.show()
plt.savefig('mask.jpg',dpi=150)


# Other choices to generate the mask are:
# 
# - **bernoulli**, each pixel in the mask is generate according to a Bernoulli random variable with probability *delta*.
# - **vardentri**, variable density triangular.
# - **vardengauss**
# - **vardenexp**
# - **halton**

# ### Sparsity in the wavelet domain
# 
# We use `pywavelets` and set the wavelet to `haar` and the padding to `zero`

# In[ ]:


waveName = 'db8'
waveMode = 'zero'


# and compute the 2D Haar wavelet transform using

# In[ ]:


import pywt
wim = pywt.coeffs_to_array(pywt.wavedec2(im, wavelet=waveName, mode=waveMode))[0]
plt.figure(figsize=(8,8))
plt.imshow(np.log(np.abs(wim)+1.0e-5), cmap='gray')
plt.axis('off')
# plt.show()
plt.savefig('wim.jpg',dpi=150)


# #### Constructing the wavelet-Fourier operator
# 
# We now define the operator that maps the wavelet coefficients of an image to the undersampled Fourier coefficients of the image.

# In[ ]:


from mrina import OperatorWaveletToFourier

# Assemble operator
A = OperatorWaveletToFourier(im.shape, samplingSet=omega[0], waveletName=waveName)
# Undersampled measurements
yim = A.eval(wim, 1)

print('Input shape: ', A.inShape)
print('Output shape: ', A.outShape)
print('Matrix shape: ', A.shape)


# ### Reconstruction from noiseless data
# 
# We now define the operator that maps the wavelet coefficients of an image to the undersampled Fourier coefficients of the image. **Warning:** This step can take about 4min.

# In[ ]:


from mrina import RecoveryL1NormNoisy

# Recovery - for low values of eta it is better to use SoS-L1Ball
wimrec_cpx, _ = RecoveryL1NormNoisy(0.01, yim, A, disp=True, method='SoS-L1Ball')
# The recovered coefficients could be complex!
imrec_cpx = A.getImageFromWavelet(wimrec_cpx)
imrec = np.abs(imrec_cpx)
# Compare true vs reconstructed image
print('Reconstruction error')
print('   Absolute ', la.norm((im - imrec_cpx).ravel()))
print('   Relative ', la.norm((im - imrec_cpx).ravel())/la.norm(im.ravel()))

# plt.imshow(im, cmap='gray')
# plt.colorbar()
# plt.title('true')
# plt.axis('off')
# plt.show()
plt.imshow(imrec, cmap='gray')
# plt.colorbar()
plt.title('recovered')
plt.axis('off')
# plt.show()
plt.savefig('imrec_noiseless.jpg',dpi=150)


# ### Noise
# 
# The measurements are usually corrupted by noise. In this case, we model it as additive Gaussian noise with variance ``sigma^2``. Usually the noise variance is determined by the **Signal-to-Noise Ratio (SNR)**. 
# 

# In[ ]:


# Target SNR
SNR = 50
# Signal power. The factor 2 accounts for real/imaginary parts
yim_pow = la.norm(yim.ravel()) ** 2 / (2 * yim.size)
# Noise st. dev.
sigma = np.sqrt(yim_pow / SNR)
# Noisy measurements
y = yim + sigma * (np.random.normal(size=yim.shape) + 1j * np.random.normal(size=yim.shape))
z_pow = la.norm((y - yim).ravel()) ** 2 / (2 * yim.size)
print('Number of complex measurements: ', yim.size)
print('Signal power: ', yim_pow)
print('Noise power:  ', z_pow)
print('SNR:          ', yim_pow/z_pow)

_ = plt.hist(np.abs((y - yim).ravel()) ** 2, bins='auto') 
plt.title("Squared deviations")
# plt.show()
plt.savefig('y.jpg',dpi=150)


# ### Noisy recovery
# 
# In this case we reconstruct the signal from noisy data. We use as parameter ``eta`` a factor ``sqrt(2 * m)``  times ``sigma``. **Warning:** This can take about 4min.
# 

# In[ ]:


# Parameter eta
eta = np.sqrt(2 * y.size) * sigma
# Recovery
wimrec_noisy_cpx, _ = RecoveryL1NormNoisy(eta, y, A, disp=True, disp_method=False, method='BPDN')
# The recovered coefficients could be complex!
imrec_noisy_cpx = A.getImageFromWavelet(wimrec_noisy_cpx)
imrec_noisy = np.abs(imrec_noisy_cpx)
# Compare true vs reconstructed image
print('Reconstruction error')
print('   Absolute ', la.norm((im - imrec_noisy_cpx).ravel()))
print('   Relative ', la.norm((im - imrec_noisy_cpx).ravel())/la.norm(im.ravel()))
# plt.imshow(im, cmap='gray')
# plt.colorbar()
# plt.title('true')
# plt.axis('off')
# plt.show()
# plt.imshow(imrec, cmap='gray')
# plt.colorbar()
# plt.title('noiseless recovery')
# plt.axis('off')
# plt.show()
plt.imshow(imrec_noisy, cmap='gray')
# plt.colorbar()
plt.title('noisy recovery')
plt.axis('off')
# plt.show()
plt.savefig('imrec_noisy.jpg',dpi=150)


# ### Debiasing
# 
# The reconstructed wavelet coefficients usually underestimate the true wavelet coefficients. To compensate for this effect, we debias the estimate. To do this, we restrict the operator to the support of the solution obtained for the noisy recovery.
# 

# In[ ]:


from mrina import lsQR

useLSQR = True

# Support of noisy solution
wim_supp = np.where(np.abs(wimrec_noisy_cpx) > 1E-4 * la.norm(wimrec_noisy_cpx.ravel(), np.inf), True, False)
# Restriction of the operator
Adeb = A.colRestrict(wim_supp)
# Solve least-squares problem
if(useLSQR):
    lsqr = lsQR(Adeb)  
    lsqr.solve(y[Adeb.samplingSet])
    wimrec_noisy_cpx_deb = np.zeros(Adeb.wavShape,dtype=np.complex)
    wimrec_noisy_cpx_deb[Adeb.basisSet] = lsqr.x[:]
else:
    wimrec_noisy_cpx_deb, _ = MinimizeSumOfSquares(y, Adeb, disp=True, printEvery=100)
# The recovered coefficients could be complex!
imrec_noisy_cpx_deb = Adeb.getImageFromWavelet(wimrec_noisy_cpx_deb)
imrec_noisy_deb = np.abs(imrec_noisy_cpx_deb)
# Compare true vs reconstructed image
print('Reconstruction error - no debiasing')
print('   Absolute ', la.norm((im - imrec_noisy_cpx).ravel()))
print('   Relative ', la.norm((im - imrec_noisy_cpx).ravel())/la.norm(im.ravel()))
print('Reconstruction error - debiasing')
print('   Absolute ', la.norm((im - imrec_noisy_cpx_deb).ravel()))
print('   Relative ', la.norm((im - imrec_noisy_cpx_deb).ravel())/la.norm(im.ravel()))
# plt.imshow(im, cmap='gray')
# plt.title('true')
# plt.axis('off')
# plt.colorbar()
# plt.show()
# plt.imshow(imrec, cmap='gray')
# plt.title('noiseless recovery')
# plt.axis('off')
# plt.colorbar()
# plt.show()
# plt.imshow(imrec_noisy, cmap='gray')
# plt.title('noisy recovery')
# plt.axis('off')
# plt.colorbar()
# plt.show()
plt.imshow(imrec_noisy_deb, cmap='gray')
plt.title('noisy recovery - debiased')
plt.axis('off')
# plt.colorbar()
# plt.show()
plt.savefig('imrec_noisy_bd.jpg',dpi=150)


# ### Stagewise Orthogonal Matching Pursuit

# In[ ]:


from mrina import OMPRecovery
# Recovery
wimrec_noisy_cpx, _ = OMPRecovery(A, y)
# The recovered coefficients could be complex!
imrec_noisy_cpx = A.getImageFromWavelet(wimrec_noisy_cpx)
imrec_noisy = np.abs(imrec_noisy_cpx)
# Compare true vs reconstructed image
print('OMP Reconstruction error')
print('   Absolute ', la.norm((im - imrec_noisy_cpx).ravel()))
print('   Relative ', la.norm((im - imrec_noisy_cpx).ravel())/la.norm(im.ravel()))
# plt.imshow(im, cmap='gray')
# plt.title('true')
# plt.axis('off')
# plt.colorbar()
# plt.show()
# plt.imshow(imrec, cmap='gray')
# plt.title('noiseless recovery')
# plt.axis('off')
# plt.colorbar()
# plt.show()
plt.imshow(imrec_noisy, cmap='gray')
plt.title('noisy recovery - STOMP')
plt.axis('off')
# plt.colorbar()
# plt.show()
plt.savefig('imrec_noisy_omp.jpg',dpi=150)


# In[ ]:




