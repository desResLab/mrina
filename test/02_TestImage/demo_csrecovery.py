import sys
sys.path.append('../')
from CSRecoverySuite import CSRecovery, Operator4dFlow, pywt2array, array2pywt, crop
import cv2
import matplotlib.pyplot as plt
import numpy.fft as fft
import pywt
import numpy as np
import os

# Show figures all together or incrementally
showAtTheEnd = False

# Load data
im   = cv2.imread('nd_small.jpg', cv2.IMREAD_GRAYSCALE)
im   = crop(im) # crop for multilevel wavelet decomp. array transform
imsz = im.shape
plt.figure()
plt.imshow(im, cmap='gray')
plt.title('true image')
if(showAtTheEnd):
  plt.draw()
else:
  plt.show()

# Wavelet Parameters
wim = pywt2array( pywt.wavedec2(im, wavelet='haar', mode='periodic'), imsz)
wsz = wim.shape

# Plot wavelet coefficients
plt.figure()
plt.imshow(wim[:, :], cmap='gray', vmin=0, vmax=np.max(np.abs(wim[:])))
if(showAtTheEnd):
  plt.draw()
else:
  plt.show()

# Create undersampling pattern
#   Sampling fraction
delta       = 0.75
#   Sampling set
omega       = np.where( np.random.uniform(0, 1, imsz) < delta, True, False )
# 4dFlow Operator
A           = Operator4dFlow( imsz=imsz, insz=wsz, samplingSet=omega, waveletName='haar', waveletMode='periodic' )
# True data (recall A takes as input wavelet coefficients)
yim         = A.eval( wim, 1 )

# Plot undersampling pattern
plt.figure();
plt.imshow(omega, cmap='gray', vmin=0, vmax=1)
plt.title('sampling set')
if(showAtTheEnd):
  plt.draw()
else:
  plt.show()

# Recovery via orthogonal projection
fim         = fft.fft2( im )
fim[~omega] = 0
fim         = fft.ifft2( fim )
plt.figure();
plt.imshow(np.absolute( fim ), cmap='gray', vmin=0, vmax=np.max( np.abs( fim ) ))
plt.title('least l2-norm reconstruction')
if(showAtTheEnd):
  plt.draw()
else:
  plt.show()

# Recovery via CSRecovery
#   Here we choose \eta as a fraction of the true image. In general it is
#   proportional to the noise variance
print('--- CS Recovery')
imNrm        = np.linalg.norm(im.ravel(), 2)
eta          = 1E-3 * imNrm
cswim, fcwim = CSRecovery(eta, yim, A, np.zeros( wsz ), disp=2, method='pgdl1')
csim         = pywt.waverec2(array2pywt( cswim ), wavelet='haar', mode='periodic')

# Covariance from adding noise with fixed undersampling pattern
stdev        = 1
noise        = np.random.normal(scale=stdev, size=im.shape) + 1j*np.random.normal(scale=stdev, size=im.shape)
fim          = fft.fft2(im) + noise
imnoise      = fft.ifft2(fim)
wim          = pywt2array( pywt.wavedec2(fft.ifft2(fim), wavelet='haar', mode='periodic'), imsz)
yim          = A.eval( wim, 1 )
cswim, fcwim = CSRecovery(eta, yim, A, np.zeros( wsz ), disp=2, method='pgdl1')
csimnoise    = pywt.waverec2(array2pywt( cswim ), wavelet='haar', mode='periodic')

cv = np.cov(noise, csimnoise-csim) #np.cov(csim, csimnoise)
plt.figure();
plt.imshow(np.abs(cv), cmap='gray')
plt.title('covariance')
if(showAtTheEnd):
  plt.draw()
else:
  plt.show()

# Summary statistics
print('l1-norm (true)', np.linalg.norm(wim.ravel(), 1))
print('l1-norm (recovered)', np.linalg.norm(cswim.ravel(), 1))
print('Reconstruction error:', np.linalg.norm((cswim - wim).ravel() , 2))
print('Residual:', np.linalg.norm((A.eval(cswim, 1) - yim).ravel() , 2))
print('Residual (true):', np.linalg.norm((A.eval(wim, 1) - yim).ravel() , 2))

plt.figure();
plt.imshow(np.abs(imnoise), cmap='gray');
plt.title('noisy true image (before undersampling)');
if(showAtTheEnd):
  plt.draw()
else:
  plt.show()

plt.figure();
plt.imshow(np.abs(imnoise-im), cmap='gray');
plt.title('diff between noisy image and original');
if(showAtTheEnd):
  plt.draw()
else:
  plt.show()

# Show recovered picture
plt.figure();
plt.imshow(np.absolute(csim), cmap='gray', vmin=0, vmax=np.linalg.norm( csim.ravel(), np.inf))
plt.title('reconstructed image')
if(showAtTheEnd):
  plt.draw()
else:
  plt.show()

# Reconstruction error
plt.figure()
plt.imshow(np.absolute(csim - im), cmap='gray')
plt.title('reconstruction error')
if(showAtTheEnd):
  plt.draw()
else:
  plt.show()

