import sys
sys.path.append('../')
import numpy as np
import matplotlib.pyplot as plt
from genSamples import add_noise
from CSRecoverySuite import generateSamplingMask
from recover import linear_reconstruction
from recover import recoverAll
from recover import recover

ipCS    = 0
ipCSDEB = 1
ipOMP   = 2

noise_percent    = 0.1
num_realizations = 2
uratio           = 0.75

# Select Method
currMethod = ipCSDEB

def computeMSEVec(rec,ref):
  res = np.zeros(rec.shape[0])
  for loopA in range(rec.shape[0]):
    #print(rec[loopA,0,0].shape,ref.shape)
    res[loopA] = np.mean((np.absolute(rec[loopA,0,0]) - np.absolute(ref))**2)
    #print(loopA,res[loopA])
  #print('ref: ',np.absolute(ref.astype(np.float))**2)
  res /= np.mean(np.absolute(ref)**2)
  return res


# Load original image
print('Reading original image...')
orimg = np.load('./ndimg/imgs_n1.npy').astype(np.float)

plt.figure(figsize=(3,3))
plt.imshow(orimg[0,0,0],cmap='gray')
plt.axis('off')
plt.title('orig')
plt.tight_layout()
plt.savefig('./orig.pdf')
plt.close()

# Transform to k-space
print('Transform in k-space...')
kspace = np.fft.fft2(orimg)

# Add noise 
print('Add noise...')
ks_samples, snr = add_noise(kspace, noise_percent, num_realizations, num_components=1)

# Add Undersampling Mask
print('Generate undesampling mask...')
omega = generateSamplingMask(ks_samples.shape[-2:], uratio, saType='vardengauss')
# omega = generateSamplingMask(ks_samples.shape[-2:], uratio, saType='bernoulli')

# Linear Reconstruction
print('Perform linear reconstruction...')
linrec = linear_reconstruction(ks_samples, omega)

plt.figure(figsize=(3,3))
plt.imshow(np.abs(linrec[0,0,0]),cmap='gray')
plt.axis('off')
plt.title('linrec')
plt.tight_layout()
plt.savefig('./linrec.pdf')
plt.close()

print('Perform nonlinear reconstruction...')
csrec = recoverAll(ks_samples, orimg, omega, noise_percent, c=2, mode=currMethod)

if(currMethod == ipCS):
  plt.figure(figsize=(3,3))
  plt.imshow(np.abs(csrec[0,0,0]),cmap='gray')
  plt.axis('off')
  plt.title('cs rec')
  plt.tight_layout()
  plt.savefig('./csrec.pdf')
  plt.close()
elif(currMethod == ipCSDEB):
  plt.figure(figsize=(3,3))
  plt.imshow(np.abs(csrec[0,0,0]),cmap='gray')
  plt.axis('off')
  plt.title('cs+deb rec')
  plt.tight_layout()
  plt.savefig('./csdebrec.pdf')
  plt.close()
elif(currMethod == ipOMP):
  plt.figure(figsize=(3,3))
  plt.imshow(np.abs(csrec[0,0,0]),cmap='gray')
  plt.axis('off')
  plt.title('omp rec')
  plt.tight_layout()
  plt.savefig('./omprec.pdf')
  plt.close()
else:
  print('ERROR: Invalid reconstruction method')
  sys.exit(-1)

# Compute Average Reconstructions
avgLin = np.mean(linrec,axis=0)
avgCS  = np.mean(csrec,axis=0)

#ax = plt.subplot(2,3,5)
#plt.imshow(np.abs(avgLin[0,0]),cmap='gray')
#plt.axis('off')
#plt.title('avgLin')

#ax = plt.subplot(2,3,6)
#plt.imshow(np.abs(avgCS[0,0]),cmap='gray')
#plt.axis('off')
#plt.title('avgCS')

#plt.tight_layout()
#plt.show()

# Compute MSE - wrt avg
mseAvgLin = computeMSEVec(linrec,avgLin[0,0])
mseAvgCS  = computeMSEVec(csrec,avgCS[0,0])
# Compute MSE - wrt orig
mseTruLin = computeMSEVec(linrec,orimg[0,0,0])
mseTruCS  = computeMSEVec(csrec,orimg[0,0,0])

print('Linear MSE wrt avg: %e' % (mseAvgLin.mean()))
print('CS MSE wrt avg: %e' % (mseAvgCS.mean()))
print('Linear MSE wrt tru: %e' % (mseTruLin.mean()))
print('CS MSE wrt tru: %e' % (mseTruCS.mean()))
