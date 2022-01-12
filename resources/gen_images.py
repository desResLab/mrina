import sys
sys.path.insert(0,'..')
import cv2
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
import pywt
import scipy
from mrina import (OMPRecovery, OperatorWaveletToFourier, RecoveryL1NormNoisy,
                   generateSamplingMask, lsQR)

# Original image
im = cv2.imread('../tests/city.png', cv2.IMREAD_GRAYSCALE)/255.0
# Rescale Image      
size_ratio = 0.75
im = cv2.resize(im,
                (0,0),
                fx=size_ratio, 
                fy=size_ratio, 
                interpolation=cv2.INTER_NEAREST)
plt.imshow(im, cmap='gray')
plt.axis('off')
plt.savefig('im.jpg',bbox_inches='tight', pad_inches=0)

# ### Generating undersampling mask in *k*-space
delta = 0.75
omega = generateSamplingMask(im.shape, delta, 'vardengauss')
nsamp = np.sum((omega == 1).ravel())/np.prod(omega.shape)
# Plot mask
plt.imshow(np.fft.fftshift(omega[0]), cmap='gray')
plt.axis('off')
plt.savefig('mask.jpg',bbox_inches='tight', pad_inches=0)

waveName = 'haar'
waveMode = 'zero'
# Normalize coefficients for plotting purposes
c_wim = pywt.wavedec2(im, wavelet=waveName, mode=waveMode)
c_wim[0] /= np.abs(c_wim[0]).max()
for detail_level in range(len(c_wim)-1):
  c_wim[detail_level + 1] = [d/np.abs(d).max() for d in c_wim[detail_level + 1]]
c_wim_plt = pywt.coeffs_to_array(c_wim)[0]
# Compute transform
wim = pywt.coeffs_to_array(pywt.wavedec2(im, wavelet=waveName, mode=waveMode))[0]
plt.imshow(c_wim_plt, cmap='seismic')
plt.axis('off')
plt.savefig('wim.jpg',bbox_inches='tight', pad_inches=0)
plt.close()

# #### Constructing the wavelet-Fourier operator
A = OperatorWaveletToFourier(im.shape, samplingSet=omega[0], waveletName=waveName)
yim = A.eval(wim, 1)

# ### Reconstruction from noiseless data
wimrec_cpx, _ = RecoveryL1NormNoisy(0.01, yim, A, disp=True, method='SoS-L1Ball')
imrec_cpx = A.getImageFromWavelet(wimrec_cpx)
imrec = np.abs(imrec_cpx)

plt.imshow(imrec, cmap='gray')
plt.axis('off')
plt.savefig('imrec_noiseless.jpg',bbox_inches='tight', pad_inches=0)
plt.close()

# ### Noise
# Target SNR
SNR = 50
yim_pow = la.norm(yim.ravel()) ** 2 / (2 * yim.size)
sigma = np.sqrt(yim_pow / SNR)
y = yim + sigma * (np.random.normal(size=yim.shape) + 1j * np.random.normal(size=yim.shape))
z_pow = la.norm((y - yim).ravel()) ** 2 / (2 * yim.size)
plt.imshow(np.fft.fftshift(np.log(np.abs(y))), cmap='gray')
plt.axis('off')
plt.savefig('y.jpg',bbox_inches='tight', pad_inches=0)
plt.close()

# ### Noisy recovery
eta = np.sqrt(2 * y.size) * sigma
wimrec_noisy_cpx, _ = RecoveryL1NormNoisy(eta, y, A, disp=True, disp_method=False, method='BPDN')
imrec_noisy_cpx = A.getImageFromWavelet(wimrec_noisy_cpx)
imrec_noisy = np.abs(imrec_noisy_cpx)

plt.imshow(imrec_noisy, cmap='gray')
plt.axis('off')
plt.savefig('imrec_noisy.jpg',bbox_inches='tight', pad_inches=0)
plt.close()

# ### Debiasing
wim_supp = np.where(np.abs(wimrec_noisy_cpx) > 1E-4 * la.norm(wimrec_noisy_cpx.ravel(), np.inf), True, False)
Adeb = A.colRestrict(wim_supp)
lsqr = lsQR(Adeb)  
lsqr.solve(y[Adeb.samplingSet])
wimrec_noisy_cpx_deb = np.zeros(Adeb.wavShape,dtype=np.complex)
wimrec_noisy_cpx_deb[Adeb.basisSet] = lsqr.x[:]
imrec_noisy_cpx_deb = Adeb.getImageFromWavelet(wimrec_noisy_cpx_deb)
imrec_noisy_deb = np.abs(imrec_noisy_cpx_deb)

plt.imshow(imrec_noisy_deb, cmap='gray')
plt.axis('off')
plt.savefig('imrec_noisy_bd.jpg',bbox_inches='tight', pad_inches=0)
plt.close()

# ### Stagewise Orthogonal Matching Pursuit
wimrec_noisy_cpx, _ = OMPRecovery(A, y)
imrec_noisy_cpx = A.getImageFromWavelet(wimrec_noisy_cpx)
imrec_noisy = np.abs(imrec_noisy_cpx)

plt.imshow(imrec_noisy, cmap='gray')
plt.axis('off')
plt.savefig('imrec_noisy_omp.jpg',bbox_inches='tight', pad_inches=0)
plt.close()