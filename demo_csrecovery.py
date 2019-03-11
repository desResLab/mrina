from CSRecoverySuite import CSRecovery, Operator4dFlow, pywt2array, array2pywt, crop
import cv2
import matplotlib.pyplot as plt
import numpy.fft as fft
import pywt
import numpy as np
import os

# Load data
im      = cv2.imread('nd_small.jpg', cv2.IMREAD_GRAYSCALE);
im = crop(im)      #crop for multilevel wavelet decomp. array transform
imsz    = im.shape;
if im.shape[0] % 2 !=0:
    newrow = np.zeros(im.shape[1])
    im = np.vstack([im, newrow])
if im.shape[1] % 2 != 0:
    newcol = np.zeros(im.shape[0])
    print(newcol.shape)
    print(np.expand_dims(newcol, axis=1).shape)
    im = np.hstack([im, np.expand_dims(newcol,axis=1)])
plt.figure();
plt.imshow(im, cmap='gray');
plt.title('true image');
plt.draw();

# Wavelet Parameters
wim         = pywt2array( pywt.wavedec2(im, wavelet='haar', mode='periodic'), imsz);
wsz         = wim.shape;
# Plot wavelet coefficients
plt.figure();
plt.imshow(wim[:, :], cmap='gray', vmin=0, vmax=np.max(np.abs(wim[:])))
plt.draw();

# Create undersampling pattern
#   Sampling fraction
delta       = 0.75;
#   Sampling set
omega       = np.where( np.random.uniform(0, 1, imsz) < delta, True, False );
# 4dFlow Operator
print(omega)
A           = Operator4dFlow( imsz=imsz, insz=wsz, samplingSet=omega, waveletName='haar', waveletMode='periodic' );
# True data (recall A takes as input wavelet coefficients)
yim         = A.eval( wim, 1 );
# Plot undersampling pattern
plt.figure();
plt.imshow(omega, cmap='gray', vmin=0, vmax=1)
plt.title('sampling set')
plt.draw()

# Recovery via orthogonal projection
fim             = fft.fft2( im );
fim[~omega]     = 0;
fim             = fft.ifft2( fim );
plt.figure();
plt.imshow(np.absolute( fim ), cmap='gray', vmin=0, vmax=np.max( np.abs( fim ) ));
plt.title('least l2-norm reconstruction')
plt.draw();

# Recovery via CSRecovery
#   Here we choose \eta as a fraction of the true image. In general it is
#   proportional to the noise variance
imNrm           = np.linalg.norm(im.ravel(), 2);
eta             = 1E-3 * imNrm;
cswim, fcwim    = CSRecovery(eta, yim, A, np.zeros( wsz ), disp=2, method='pgdl1');
csim            = pywt.waverec2(array2pywt( cswim ), wavelet='haar', mode='periodic');

# Summary statistics
print('l1-norm (true)', np.linalg.norm(wim.ravel(), 1))
print('l1-norm (recovered)', np.linalg.norm(cswim.ravel(), 1))
print('Reconstruction error:', np.linalg.norm((cswim - wim).ravel() , 2))
print('Residual:', np.linalg.norm((A.eval(cswim, 1) - yim).ravel() , 2))
print('Residual (true):', np.linalg.norm((A.eval(wim, 1) - yim).ravel() , 2))

# Show recovered picture
plt.figure();
plt.imshow(np.absolute(csim), cmap='gray', vmin=0, vmax=np.linalg.norm( csim.ravel(), np.inf))
plt.title('reconstructed image')
plt.draw()

# Reconstruction error
plt.figure()
plt.imshow(np.absolute( csim - im ), cmap='gray')
plt.title('reconstruction error')
plt.show()
