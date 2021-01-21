import sys
import numpy as np
import matplotlib.pyplot as plt

rec = np.load('rec_noise0_p75vardengauss_n2.npy')
print('rec ',rec.shape)

# Get Min and Max


noise = np.load('noisy_noise0_n2.npy')
print('noise ',noise.shape)

mask = np.load('undersamplpattern_p75vardengauss.npy')
print('mask ',mask.shape)
exit()

# pts = np.load('points_s100_n50.npy')
pts = np.load('points_s100_n50_fluidmask.npy')
img = np.load('imgs_n1.npy')

print(pts.shape)

# Loop on the distances

for loopA in range(pts.shape[0]):
  plt.imshow(img[0,0,0],alpha=0.5,cmap='gray')
  for loopB in range(pts.shape[1]):
    plt.plot(pts[loopA,loopB,:,1],pts[loopA,loopB,:,0],'bo-',alpha=0.5)

  plt.show()