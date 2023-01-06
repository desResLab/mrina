import numpy as np
import matplotlib.pyplot as plt

def compute_avg_ratio(imag,mask):

  # Get maximum and minumum
  v1 = np.abs(imag[0,1,0,mask])
  v2 = np.abs(imag[0,2,0,mask])
  v3 = np.abs(imag[0,3,0,mask])
  vmod = v1+v2+v3

  # Compute average ratio
  avg_ratio_1 = np.mean(np.abs(v1/vmod)*100)
  avg_ratio_2 = np.mean(np.abs(v2/vmod)*100)
  avg_ratio_3 = np.mean(np.abs(v3/vmod)*100)

  return avg_ratio_1,avg_ratio_2,avg_ratio_3

# MAIN
if __name__ == "__main__":

  imag = np.load('sl1_imgs_n1.npy')
  mask = np.load('sl1_mask.npy')

  ar1,ar2,ar3 = compute_avg_ratio(imag,mask)

  # Print info
  print('SL1')
  print('Average ratio for component 1: ',ar1)
  print('Average ratio for component 2: ',ar2)
  print('Average ratio for component 3: ',ar3)
  print('')

  imag = np.load('sl2_imgs_n1.npy')
  mask = np.load('sl2_mask.npy')

  ar1,ar2,ar3 = compute_avg_ratio(imag,mask)

  # Print info
  print('SL2')
  print('Average ratio for component 1: ',ar1)
  print('Average ratio for component 2: ',ar2)
  print('Average ratio for component 3: ',ar3)
  print('')

  imag = np.load('sl3_imgs_n1.npy')
  mask = np.load('sl3_mask.npy')

  ar1,ar2,ar3 = compute_avg_ratio(imag,mask)

  # Print info
  print('SL3')
  print('Average ratio for component 1: ',ar1)
  print('Average ratio for component 2: ',ar2)
  print('Average ratio for component 3: ',ar3)
  print('')

