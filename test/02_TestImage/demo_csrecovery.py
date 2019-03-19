import sys
sys.path.append('../../')
from CSRecoverySuite import CSRecovery, Operator4dFlow, pywt2array, array2pywt, crop
import cv2
import matplotlib.pyplot as plt
import numpy.fft as fft
import pywt
import numpy as np
import math
import os
from multiprocessing import Process, cpu_count, Manager
home = os.getenv('HOME')

# Show figures all together or incrementally
showAtTheEnd = False
num_samples = 100
noise_percent = 0.1
dir = home + "/Documents/undersampled/npy/"
generate = False #whether to generate noisy realizations or not
csrecover=True #whether to recover the noisy realizations or not

def add_noise(im, imNrm, num_samples,noise_percent):
    avgnorm = imNrm/math.sqrt(im.size)
    stdev = noise_percent * avgnorm
    samples = np.zeros((num_samples,) + imsz, dtype=complex)
    for n in range(num_samples):
        noise        = np.random.normal(scale=stdev, size=im.shape) + 1j*np.random.normal(scale=stdev, size=im.shape)
        fim          = fft.fft2(im) + noise
        imnoise      = fft.ifft2(fim)
        samples[n] = imnoise
    np.save(dir + 'ndimg_noise'+str(int(noise_percent*100))+'_n'+str(num_samples), samples)

def recover(imgs,eta, A, processnum, return_dict):
    def recover(im,imsz,eta):
        wim          = pywt2array( pywt.wavedec2(im, wavelet='haar', mode='periodic'), imsz)
        yim          = A.eval( wim, 1 )
        wsz = wim.shape
        cswim, fcwim = CSRecovery(eta, yim, A, np.zeros( wsz ), disp=0, method='pgdl1')
        csim    = pywt.waverec2(array2pywt( cswim ), wavelet='haar', mode='periodic')
        return csim
    cs = np.zeros(imgs.shape,dtype=complex)
    imsz = imgs.shape[1:]
    for k in range(imgs.shape[0]):
        cs[k] = recover(imgs[k], imsz, eta)
    return_dict[processnum] = cs
    return cs

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
#cswim, fcwim = CSRecovery(eta, yim, A, np.zeros( wsz ), disp=2, method='pgdl1')
#csim         = pywt.waverec2(array2pywt( cswim ), wavelet='haar', mode='periodic')

# Covariance from adding noise with fixed undersampling pattern
if generate:
    add_noise(im,imNrm,num_samples,noise_percent)

s = np.load(dir + 'ndimg_noise'+str(int(noise_percent*100))+'_n'+str(num_samples) + '.npy')
print(cpu_count())
c = cpu_count()-2
if csrecover:
    np.save(dir + 'ndpattern_noise'+str(int(noise_percent*100))+'_n'+str(num_samples), omega)
    interval = max(int(s.shape[0]/c),1)
    manager = Manager()
    return_dict = manager.dict()
    jobs = []
    A = Operator4dFlow( imsz=imsz, insz=wsz, samplingSet=omega, waveletName='haar', waveletMode='periodic' )
    for n in range(0, s.shape[0], interval):
        p = Process(target=recover, args=(s[n:n+interval], eta, A, int(n/interval), return_dict))
        jobs.append(p)
        p.start()
    for job in jobs:
        job.join()
    print(return_dict.values())
    np.save(dir + 'ndrec_noise'+str(int(noise_percent*100))+'_n'+str(num_samples), np.asarray(return_dict.values()))
    print('finish')

plt.figure();
plt.imshow(np.abs(return_dict[0][0]), cmap='gray');
plt.title('recovered');
if(showAtTheEnd):
  plt.draw()
else:
  plt.show()
imnoise=s[0]
wim          = pywt2array( pywt.wavedec2(imnoise, wavelet='haar', mode='periodic'), imsz)
yim          = A.eval( wim, 1 )
cswim, fcwim = CSRecovery(eta, yim, A, np.zeros( wsz ), disp=2, method='pgdl1')
csimnoise    = pywt.waverec2(array2pywt( cswim ), wavelet='haar', mode='periodic')


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
plt.imshow(np.absolute(csimnoise), cmap='gray', vmin=0, vmax=np.linalg.norm( csimnoise.ravel(), np.inf))
plt.title('reconstructed image')
if(showAtTheEnd):
  plt.draw()
else:
  plt.show()

# Reconstruction error
plt.figure()
plt.imshow(np.absolute(csimnoise - im), cmap='gray')
plt.title('reconstruction error')
if(showAtTheEnd):
  plt.draw()
else:
  plt.show()
