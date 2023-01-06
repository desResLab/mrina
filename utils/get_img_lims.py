import numpy as np
import matplotlib.pyplot as plt

"""
Script to print the maximum and minum value
of every component of the original image, i.e., k_i,i={0,1,2,3}.
The limits are then used for the mrina.save_imgs command
"""

imag = np.load('imgs_n1.npy')
mask = np.load('sl_mask.npy')

# im_mask = imag.copy()
# im_mask[0,0,0] = 0.0

# Get maximum and minumum
print('Min k0: ',imag[0,0,0,mask].min())
print('Max k0: ',imag[0,0,0,mask].max())
print('')
print('Min k1: ',imag[0,1,0,mask].min())
print('Max k1: ',imag[0,1,0,mask].max())
print('')
print('Min k2: ',imag[0,2,0,mask].min())
print('Max k2: ',imag[0,2,0,mask].max())
print('')
print('Min k3: ',imag[0,3,0,mask].min())
print('Max k3: ',imag[0,3,0,mask].max())



