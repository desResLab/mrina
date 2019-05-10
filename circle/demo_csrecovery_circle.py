from CSRecoverySuite import crop, CSRecovery, Operator4dFlow, pywt2array, array2pywt, CSRecoveryDebiasing
import cv2
import matplotlib.pyplot as plt
import numpy.fft as fft
import pywt
import numpy as np
import os

# Load data
im      = cv2.imread('circle.jpg', cv2.IMREAD_GRAYSCALE)
im      = crop(im) # Crop for multilevel wavelet decomp. array transform
imsz    = im.shape
print('Image size:', imsz)

# Wavelet Parameters
wim         = pywt2array(pywt.wavedec2(im, wavelet='haar', mode='periodic'), imsz);
wsz         = wim.shape
print('Non-zero coefficients:', np.sum(np.where(np.absolute(wim.ravel()) > 0, 1, 0)))
print('Non-zero fraction:', np.sum(np.where(np.absolute(wim.ravel()) > 0, 1, 0)) / np.prod(imsz))
# Create undersampling pattern
#   Sampling fraction
delta       = 0.5;
#   Sampling set
omega       = np.where( np.random.uniform(0, 1, imsz) < delta, True, False );
# 4dFlow Operator
A           = Operator4dFlow( imsz=imsz, insz=wsz, samplingSet=omega, waveletName='haar', waveletMode='periodic' );
# True data (recall A takes as input wavelet coefficients)
yim         = A.eval( wim, 1 );

# Recovery via orthogonal projection
fim             = fft.fft2( im );
fim[~omega]     = 0;
fim             = fft.ifft2( fim );

# Recovery via CSRecovery
#   Here we choose \eta as a fraction of the true image. In general it is
#   proportional to the noise variance
imNrm           = np.linalg.norm(im.ravel(), 2);
eta             = 5E1;
cswim, fcwim    = CSRecovery(eta, yim, A, np.zeros( wsz ), disp=3);
csim            = pywt.waverec2(array2pywt(cswim), wavelet='haar', mode='periodic');
deb_cswim, deb_fcwim  = CSRecoveryDebiasing(yim, A, cswim)
deb_csim        = pywt.waverec2(array2pywt(deb_cswim), wavelet='haar', mode='periodic');


# Summary statistics
print('l1-norm (true)', np.linalg.norm(wim.ravel(), 1))
print('l1-norm (recovered)', np.linalg.norm(cswim.ravel(), 1))
print('Reconstruction error:', np.linalg.norm((cswim - wim).ravel() , 2))
print('Reconstruction error (debiased):', np.linalg.norm((deb_cswim - wim).ravel() , 2))
print('Residual / eta:', np.linalg.norm((A.eval(cswim, 1) - yim).ravel() , 2), eta)
print('Residual (true):', np.linalg.norm((A.eval(wim, 1) - yim).ravel() , 2))
print('Residual (debiased):', np.linalg.norm((A.eval(deb_cswim, 1) - yim).ravel() , 2))

# Show recovered picture
plt.figure();
plt.imshow(np.absolute(csim), cmap='gray', vmin=0, vmax=np.linalg.norm( csim.ravel(), np.inf))
plt.title('reconstructed image')
plt.draw()

# Show debiased picture
plt.figure();
plt.imshow(np.absolute(deb_csim), cmap='gray', vmin=0, vmax=np.linalg.norm( csim.ravel(), np.inf))
plt.title('reconstructed image')
plt.draw()

# Reconstruction error
plt.figure()
plt.imshow(np.absolute( csim - im ), cmap='gray')
plt.title('reconstruction error')
plt.draw()
