import os,sys
sys.path.append('../../')
from CSRecoverySuite import crop, CSRecovery, OperatorNorm, OperatorTestAdjoint, Operator4dFlow, pywt2array, array2pywt, CSRecoveryDebiasing, OMPRecovery
from CSRecoverySuite import generateSamplingMask
import cv2
import matplotlib.pyplot as plt
import numpy.fft as fft
import pywt
import numpy as np

# Read command-line parameters
if(len(sys.argv) < 4):
  print('Recontruction of a complex image with various approaches:')
  print('- CS Iterative thresholding')
  print('- CS Iterative thresholding + Debiasing')
  print('- Orthogonal Matching Pursuit')
  print('')
  print('usage: python3 csrecovery.py fileName uRatio NoiseInt')
  print('where:')
  print('fileName - name of the image file. Both image dimensions should be ')
  print('           a power of 2 for orthogonality in the wavelet transform.')
  print('uRatio   - undersampling ratio. This is the ratio of the available ')
  print('           k-space frequency measurements.')
  print('NoiseInt - noise intensity. This is used to set the noise standard ')
  print('           deviation to sigma = NoiseInt * imNrmAvg, where imNrmAvg')
  print('           is the average norm in image space.')
  sys.exit(-1)
else:
  fileName  = sys.argv[1]
  deltaVal  = float(sys.argv[2])
  nlevelVal = float(sys.argv[3])

# Set Wavelet Padding Mode
waveMode = 'periodization'

# Load data
im      = cv2.imread(fileName, cv2.IMREAD_GRAYSCALE)
im      = crop(im) # Crop for multilevel wavelet decomp. array transform
imsz    = im.shape
print('Image size:', imsz)

# Wavelet Parameters
wim         = pywt2array(pywt.wavedec2(im, wavelet='haar', mode=waveMode), imsz)
wsz         = wim.shape
print('Non-zero coefficients:', np.sum(np.where(np.absolute(wim.ravel()) > 0, 1, 0)))
print('Non-zero fraction:', np.sum(np.where(np.absolute(wim.ravel()) > 0, 1, 0)) / np.prod(imsz))

# Create undersampling pattern
#   Sampling fraction (for Bernoulli sampling)
delta = deltaVal
omega = generateSamplingMask(imsz, delta, 'gaussian')

plt.imshow(np.absolute(np.fft.fftshift(omega)), cmap='gray', vmin=0, vmax=1)
plt.show()

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
nlevel      = nlevelVal
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

print('')

print('--- CS Reconstruction...')
# --- Recovery via CS
cswim, fcwim    = CSRecovery(eta, yim, A, np.zeros( wsz ), disp=3)
csim            = pywt.waverec2(array2pywt(cswim), wavelet='haar', mode=waveMode)
print('l1-norm (true)', np.linalg.norm(wim.ravel(), 1))
print('l1-norm (recovered)', np.linalg.norm(cswim.ravel(), 1))
print('Reconstruction error:', np.linalg.norm((cswim - wim).ravel() , 2))
print('Residual (true):', np.linalg.norm((A.eval(wim, 1) - yim).ravel() , 2))
print('Residual (recovered):', np.linalg.norm((A.eval(cswim, 1) - yim).ravel() , 2), eta)

print('')

print('--- CS-Debiasing Reconstruction...')
# --- Debiasing CS-Debiasing  
deb_cswim, deb_fcwim  = CSRecoveryDebiasing(yim, A, cswim)
deb_csim        = pywt.waverec2(array2pywt(deb_cswim), wavelet='haar', mode=waveMode)
print('l1-norm (recovered+debiased)', np.linalg.norm(deb_cswim.ravel(), 1))
print('Reconstruction error (debiased):', np.linalg.norm((deb_cswim - wim).ravel() , 2))
print('Residual (recovered+debiased):', np.linalg.norm((A.eval(deb_cswim, 1) - yim).ravel() , 2))

print('')

print('--- OMP Reconstruction...')
# --- OMP reconstruction
omp_cswim, omp_fcwim  = OMPRecovery(A, yim, tol=5.0e-2)
omp_csim = pywt.waverec2(array2pywt(omp_cswim), wavelet='haar', mode=waveMode)
# Summary statistics
print('l1-norm (OMP)', np.linalg.norm(omp_cswim.ravel(), 1))
print('Reconstruction error (OMP):', np.linalg.norm((omp_cswim - wim).ravel() , 2))
print('Residual (OMP):', np.linalg.norm((A.eval(omp_cswim, 1) - yim).ravel() , 2))
# Show OMP picture and reconstruction error

plt.figure(figsize=(10,8))
# CS
plt.subplot(3,3,1)
plt.imshow(np.absolute(im), cmap='gray', vmin=0, vmax=np.linalg.norm( im.ravel(), np.inf))
plt.title('original image')
plt.axis('off')
plt.subplot(3,3,2)
plt.imshow(np.absolute(csim), cmap='gray', vmin=0, vmax=np.linalg.norm( csim.ravel(), np.inf))
plt.title('reconstructed image (CS)')
plt.axis('off')
plt.subplot(3,3,3)
plt.imshow(np.absolute(csim-im), cmap='gray')
plt.title('reconstruction error (CS)')
plt.axis('off')
# CS+Debias
plt.subplot(3,3,4)
plt.imshow(np.absolute(im), cmap='gray', vmin=0, vmax=np.linalg.norm( im.ravel(), np.inf))
plt.title('original image')
plt.axis('off')
plt.subplot(3,3,5)
plt.imshow(np.absolute(deb_csim), cmap='gray', vmin=0, vmax=np.linalg.norm( deb_csim.ravel(), np.inf))
plt.title('reconstructed image (CS+debiased)')
plt.axis('off')
plt.subplot(3,3,6)
plt.imshow(np.absolute(deb_csim-im), cmap='gray')
plt.title('reconstruction error (CS+debiased)')
plt.axis('off')
# OMP
plt.subplot(3,3,7)
plt.imshow(np.absolute(im), cmap='gray', vmin=0, vmax=np.linalg.norm( im.ravel(), np.inf))
plt.title('original image')
plt.axis('off')
plt.subplot(3,3,8)
plt.imshow(np.absolute(omp_csim), cmap='gray', vmin=0, vmax=np.linalg.norm( omp_csim.ravel(), np.inf))
plt.title('reconstructed image (OMP)')
plt.axis('off')
plt.subplot(3,3,9)
plt.imshow(np.absolute(omp_csim-im), cmap='gray')
plt.title('reconstruction error (OMP)')
plt.axis('off')

plt.tight_layout()
plt.show()
