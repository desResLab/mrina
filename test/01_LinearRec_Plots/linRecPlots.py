import matplotlib.pyplot as plt
import numpy.fft as fft
import numpy as np
import math

# image size
n           = 512
imsz        = (n, n)

# sampling ratio
delta       = 0.001
Omega       = np.where( np.random.uniform(0, 1, imsz) < delta, True, False )
FX          = np.zeros(( n, n ))
FX[ Omega ] = 1 / np.count_nonzero( Omega )
X           = n**2 * fft.fftshift( fft.ifft2( FX ) )

fs=8
plt.rc('font',  family='serif')
plt.rc('xtick', labelsize='x-small')
plt.rc('ytick', labelsize='x-small')
plt.rc('text',  usetex=True)

plt.figure(figsize=(4,3))
plt.imshow(np.absolute( Omega ), cmap='gray', vmin=0, vmax=1/np.count_nonzero( Omega ))
plt.xlabel('Width',fontsize=fs)
plt.ylabel('Height',fontsize=fs)
plt.tick_params(labelsize=fs)
plt.savefig('linRec_01_omega.pdf')

plt.figure(figsize=(4,3))
plt.imshow(np.absolute( X ), cmap='jet', vmin=0, vmax=1, aspect='equal')
plt.xlabel('Width',fontsize=fs)
plt.ylabel('Height',fontsize=fs)
plt.tick_params(labelsize=fs)
plt.savefig('linRec_01_x.pdf')

# sampling ratio
delta       = 0.0001
Omega       = np.where( np.random.uniform(0, 1, imsz) < delta, True, False )
FX          = np.zeros(( n, n ))
FX[ Omega ] = 1 / np.count_nonzero( Omega )
X           = n**2 * fft.fftshift( fft.ifft2( FX ) )

plt.figure(figsize=(4,3))
plt.imshow(np.absolute( Omega ), cmap='gray', vmin=0, vmax=1/np.count_nonzero( Omega ))
plt.colorbar()
plt.xlabel('Width',fontsize=fs)
plt.ylabel('Height',fontsize=fs)
plt.tick_params(labelsize=fs)
plt.savefig('linRec_02_omega.pdf')

plt.figure(figsize=(4,3))
plt.imshow(np.absolute( X ), cmap='jet', vmin=0, vmax=1, aspect='equal')
plt.colorbar()
plt.xlabel('Width',fontsize=fs)
plt.ylabel('Height',fontsize=fs)
plt.tick_params(labelsize=fs)
plt.savefig('linRec_02_x.pdf')
