# %%
import sys
import numpy as np
import numpy.linalg as la

sys.path.append('../')

# %%
import cv2
import matplotlib.pyplot as plt

im = cv2.imread('city.png', cv2.IMREAD_GRAYSCALE)
plt.imshow(im, cmap='gray')
plt.axis('off')
print('Image size: ', im.shape)

# %%
from CSRecoverySuite import generateSamplingMask
# Select an undesampling ratio
delta = 0.25
# Generate an undersampling mask
omega = generateSamplingMask(im.shape, delta, 'bernoulli')
# Verify the undersampling ratio
nsamp = np.sum((omega == 1).ravel())/np.prod(omega.shape)
print('Effective undersampling', nsamp)
# Plot mask
plt.imshow(omega[0], cmap='binary')
plt.axis('off')
plt.show()
# %%
import pywt

waveName = 'haar'
waveMode = 'zero'

wim = pywt.coeffs_to_array(pywt.wavedec2(im, wavelet=waveName, mode=waveMode))[0]
plt.imshow(wim, cmap='gray')
plt.axis('off')
plt.show()

# %%
from maps import OperatorWaveletToFourier

A = OperatorWaveletToFourier(im.shape, samplingSet=omega[0], waveletName=waveName)
# Undersampled measurements
yim = A.eval(wim, 1)

print('Input shape ', A.inShape)
print('Output shape ', A.outShape)
print('Matrix shape ', A.shape)

# %%
from solver_l1_norm import RecoveryL1NormNoisy

# Recovery
wimrec, _ = RecoveryL1NormNoisy(0.01, yim, A, disp=True, disp_method=False, method='SoS-L1Ball')
# The recovered coefficients could be complex!
imrec_cpx = A.getImageFromWavelet(wimrec)
imrec = np.abs(imrec_cpx)
# Compare true vs reconstructed image
print('Reconstruction error')
print('   Absolute ', la.norm((im - imrec_cpx).ravel()))
print('   Relative ', la.norm((im - imrec_cpx).ravel())/la.norm(im.ravel()))
# %%
# Target SNR
SNR = 5
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

# %%
# Parameter eta
from solver_l1_norm import RecoveryL1NormNoisy

eta = np.sqrt(2 * y.size) * sigma
print('eta:', eta)
# Recovery
wimrec, _ = RecoveryL1NormNoisy(eta, y, A, disp=True, disp_method=True, method='BPDN')
# The recovered coefficients could be complex!
imrec_cpx_noisy = A.getImageFromWavelet(wimrec)
imrec_noisy = np.abs(imrec_cpx)
# Compare true vs reconstructed image
print('Reconstruction error')
print('   Absolute ', la.norm((im - imrec_cpx_noisy).ravel()))
print('   Relative ', la.norm((im - imrec_cpx_noisy).ravel())/la.norm(im.ravel()))