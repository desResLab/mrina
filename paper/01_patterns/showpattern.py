import sys
# sys.path.append('../../')
sys.path.append('/home/dschiava/Documents/01_Development/01_PythonApps/03_Reconstruction/')
import numpy as np
import matplotlib.pyplot as plt
from CSRecoverySuite import generateSamplingMask

fname = 'undersamplpattern_p25vardengauss_n100.npy'
# fname = 'undersamplpattern_p50vardengauss_n100.npy'
# fname = 'undersamplpattern_p75bernoulli_n100.npy'
# fname = 'undersamplpattern_p75vardengauss_n100.npy'
# fname = 'undersamplpattern_p80vardengauss_n100.npy'
# fname = 'undersamplpattern_p85vardengauss_n100.npy'
# fname = 'undersamplpattern_p90vardengauss_n100.npy'
# fname = 'undersamplpattern_p95vardengauss_n100.npy'

plt.figure(figsize=(6,3))
ax = plt.subplot(1,2,1)
img = np.load(fname)
if(len(img.shape)==3):
  ax.imshow(img[0], cmap='gray')
  imsz = img[0].shape
else:
  ax.imshow(img, cmap='gray')
  imsz = img.shape

ax = plt.subplot(1,2,2)
mask = generateSamplingMask(imsz, 0.75, saType='vardengauss')
ax.imshow(mask[0], cmap='gray')


# 'vardentri':
    # elif saType =='vardengauss': #gaussian density
    # elif saType == 'vardenexp': #exponential density
    # elif saType == 'halton': #halton sequence


plt.show()




