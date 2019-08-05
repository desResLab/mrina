import os,sys
from CSRecoverySuite import crop, CSRecovery, OperatorNorm, OperatorTestAdjoint, Operator4dFlow, pywt2array, array2pywt, CSRecoveryDebiasing, OMPRecovery
import cv2
import matplotlib.pyplot as plt
import numpy.fft as fft
import pywt
import numpy as np


# Running Modes
recCS       = False
recCSDebias = False
recOMP      = True

# Set Wavelet Padding Mode
waveMode = 'periodization'

# Load data
# im      = cv2.imread('circle.jpg', cv2.IMREAD_GRAYSCALE)
im      = cv2.imread('circle_red.jpg', cv2.IMREAD_GRAYSCALE)
im      = im[0::2, 0::2]
im      = crop(im) # Crop for multilevel wavelet decomp. array transform
imsz    = im.shape
print('Image size:', imsz)

# Wavelet Parameters
wim         = pywt2array(pywt.wavedec2(im, wavelet='haar', mode=waveMode), imsz)
wsz         = wim.shape
print('Non-zero coefficients:', np.sum(np.where(np.absolute(wim.ravel()) > 0, 1, 0)))
print('Non-zero fraction:', np.sum(np.where(np.absolute(wim.ravel()) > 0, 1, 0)) / np.prod(imsz))

# Create undersampling pattern
#   Sampling fraction
delta       = 0.6
#   Sampling set
omega       = np.where(np.random.uniform(0, 1, imsz) < delta, True, False)
nsamp       = np.sum(np.where( omega, 1, 0 ).ravel())
print('Samples:', nsamp)
print('Sampling ratio:', nsamp/im.size)

# 4D-Flow Operator
A = Operator4dFlow(imsz=imsz, insz=wsz, samplingSet=omega, waveletName='haar', waveletMode=waveMode)
print('Operator norm:', A.getNorm())

# Test if operator is adjoint
OperatorTestAdjoint(A)

# True data (recall A takes as input wavelet coefficients)
yim         = A.eval(wim, 1)

# Noisy data
nlevel      = 0.1
imNrm       = np.linalg.norm(im.ravel(), ord=2)
imNrmAvg    = imNrm / np.sqrt(2 * im.size)
wimNrm      = np.linalg.norm(wim.ravel(), ord=2)
wimNrmAvg   = wimNrm / np.sqrt(2 * wim.size)
sigma       = nlevel * (imNrm / np.sqrt(2 * im.size))
y           = yim + sigma * (np.random.normal(size=yim.shape) + 1j * np.random.normal(size=yim.shape))

# Write Messages
print('Largest image entry:', np.max(im.ravel()))
print('Largest magnitude wavelet coefficient:', np.max(wim.ravel()))
print('Image root mean-squared:', imNrmAvg)
print('Wavelet root mean-squared:', wimNrmAvg)
print('Norm of error:', np.linalg.norm(y - yim, ord=2))
print('Error estimate:', sigma * np.sqrt(2 * nsamp))
print('Noise variance:', sigma)
print('PSNR', imNrmAvg ** 2 / sigma ** 2 )
print('PSNR (dB)', 10 * np.log10(imNrmAvg ** 2 / sigma ** 2) )

# Recovery via orthogonal projection
#   FOURIER TRANSFORM NEEDS TO BE NORMALIZED
fim             = (im.size ** -1/2) * fft.fft2( im )
fim[~omega]     = 0
fim             = (im.size ** -1/2) * fft.ifft2( fim )

# Recovery via CSRecovery
#   Here we choose \eta as a fraction of the true image. In general it is
#   proportional to the noise variance
eta             = sigma * (np.sqrt(2 * nsamp) + 1.6)
print('Error bound (eta):', eta)
print('Norm of error:', np.linalg.norm(y - yim, ord=2))

if(recCS):
  print('--- CS Reconstruction...')
  # --- Recovery via CS
  cswim, fcwim    = CSRecovery(eta, yim, A, np.zeros( wsz ), disp=3)
  csim            = pywt.waverec2(array2pywt(cswim), wavelet='haar', mode=waveMode)
  print('l1-norm (true)', np.linalg.norm(wim.ravel(), 1))
  print('l1-norm (recovered)', np.linalg.norm(cswim.ravel(), 1))
  print('Reconstruction error:', np.linalg.norm((cswim - wim).ravel() , 2))
  print('Residual (true):', np.linalg.norm((A.eval(wim, 1) - yim).ravel() , 2))
  print('Residual (recovered):', np.linalg.norm((A.eval(cswim, 1) - yim).ravel() , 2), eta)
  # Show recovered picture and CS Reconstruction error
  plt.figure(figsize=(10,3))
  plt.subplot(1,3,1)
  plt.imshow(np.absolute(im), cmap='gray', vmin=0, vmax=np.linalg.norm( im.ravel(), np.inf))
  plt.title('original image')
  plt.subplot(1,3,2)
  plt.imshow(np.absolute(csim), cmap='gray', vmin=0, vmax=np.linalg.norm( csim.ravel(), np.inf))
  plt.title('reconstructed image')
  plt.subplot(1,3,3)
  plt.imshow(np.absolute(csim-im), cmap='gray')
  plt.title('reconstruction error')
  plt.tight_layout()
  plt.show()

if(recCSDebias):
  print('--- CS-Debiasing Reconstruction...')
  # --- Debiasing CS-Debiasing  
  deb_cswim, deb_fcwim  = CSRecoveryDebiasing(yim, A, cswim)
  deb_csim        = pywt.waverec2(array2pywt(deb_cswim), wavelet='haar', mode=waveMode)
  print('l1-norm (recovered+debiased)', np.linalg.norm(deb_cswim.ravel(), 1))
  print('Reconstruction error (debiased):', np.linalg.norm((deb_cswim - wim).ravel() , 2))
  print('Residual (recovered+debiased):', np.linalg.norm((A.eval(deb_cswim, 1) - yim).ravel() , 2))
  # Show debiased picture
  plt.figure()
  plt.imshow(np.absolute(deb_csim), cmap='gray', vmin=0, vmax=np.linalg.norm( deb_csim.ravel(), np.inf))
  plt.title('reconstructed image (debiased)')
  plt.show()
  # CS-Debiased Reconstruction error
  plt.figure()
  plt.imshow(np.absolute(deb_csim-im), cmap='gray')
  plt.title('reconstruction error (debiased)')
  plt.show()

if(recOMP):
  print('--- OMP Reconstruction...')
  # --- OMP reconstruction
  omp_cswim, omp_fcwim  = OMPRecovery(A, yim, tol=5.0e-2)
  omp_cswim = omp_cswim.reshape(imsz)
  omp_csim = pywt.waverec2(array2pywt(omp_cswim), wavelet='haar', mode=waveMode)
  # Summary statistics
  print('l1-norm (OMP)', np.linalg.norm(omp_cswim.ravel(), 1))
  print('Reconstruction error (OMP):', np.linalg.norm((omp_cswim - wim).ravel() , 2))
  print('Residual (OMP):', np.linalg.norm((A.eval(omp_cswim, 1) - yim).ravel() , 2))
  # Show OMP picture and reconstruction error
  plt.figure(figsize=(10,3))
  plt.subplot(1,3,1)
  plt.imshow(np.absolute(im), cmap='gray', vmin=0, vmax=np.linalg.norm( im.ravel(), np.inf))
  plt.title('original image')
  plt.subplot(1,3,2)
  plt.imshow(np.absolute(omp_csim), cmap='gray', vmin=0, vmax=np.linalg.norm( omp_csim.ravel(), np.inf))
  plt.title('reconstructed image (OMP)')
  plt.subplot(1,3,3)
  plt.imshow(np.absolute(omp_csim-im), cmap='gray')
  plt.title('reconstruction error (OMP)')
  plt.tight_layout()
  plt.show()
