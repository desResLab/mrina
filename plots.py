import matplotlib.pyplot as plt
import numpy.fft as fft
import numpy as np
import math

# image size
n           = 512;
imsz        = (n, n);
# sampling ratio
delta       = 0.001;
Omega       = np.where( np.random.uniform(0, 1, imsz) < delta, True, False );
FX          = np.zeros(( n, n ));
FX[ Omega ] = 1 / np.count_nonzero( Omega );
X           = math.pow( n, 2 ) * fft.fftshift( fft.ifft2( FX ) );
plt.figure();
plt.imshow(np.absolute( Omega ), cmap='gray', vmin=0, vmax=1/np.count_nonzero( Omega ));
plt.imshow(np.absolute( X ), cmap='jet', vmin=0, vmax=1, aspect='equal');

# sampling ratio
delta       = 0.0001;
Omega       = np.where( np.random.uniform(0, 1, imsz) < delta, True, False );
FX          = np.zeros(( n, n ));
FX[ Omega ] = 1 / np.count_nonzero( Omega );
X           = math.pow( n, 2 ) * fft.fftshift( fft.ifft2( FX ) );

plt.figure();
plt.imshow(np.absolute( Omega ), cmap='gray', vmin=0, vmax=1/np.count_nonzero( Omega ));
plt.imshow(np.absolute( X ), cmap='jet', vmin=0, vmax=1, aspect='equal');
