import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def eval_snr(img_file,signal_mask_file,noise_mask_file):
  """
  Computes the signal to noise ratio of density and velocity images
  inputs: img_file        : npy image file with shape (0,4,0,pixels_x,pixels_y)
          signal_mask_file: npy boolean 2D mask containing the interior of the fluid region (signal)
          noise_mask_file : npy boolean 2D mask containing an area outside the signal region. 
  """

  print('--- Files')
  print('Image file: ',img_file)
  print('Signal mask file: ',signal_mask_file)
  print('Noise mask file: ',noise_mask_file)

  img         = np.load(img_file)
  signal_mask = np.load(signal_mask_file)
  noise_mask  = np.load(noise_mask_file)

  img_density    = img[0,0,0]
  signal_density = np.mean(img[0,0,0][signal_mask])
  noise_density  = np.std(img[0,0,0][noise_mask])/0.66
  snr_density    = signal_density/noise_density
  perc_density   = (noise_density/signal_density)*100

  img_vel    = np.linalg.norm(img[0,1:,0],axis=0)
  signal_vel = np.mean(img_vel[signal_mask])
  noise_vel  = np.std(img_vel[noise_mask])/0.66
  snr_vel    = signal_vel/noise_vel
  perc_vel   = (noise_vel/signal_vel)*100

  print('--- Density Image')
  print('Signal: ',signal_density)  
  print('Noise variability: ',noise_density)  
  print('SNR of density image: ',snr_density)
  print('Noice percent value for density: ',perc_density)
  print('--- Velocity Image')
  print('Signal: ',signal_vel)  
  print('Noise variability: ',noise_vel)  
  print('SNR of velocity image: ',snr_vel)
  print('Noice percent value for velocity: ',perc_vel)
  print('')

# ====
# MAIN
# ====
if __name__ == "__main__":

  img_file         = 'imgs_n1_01.npy'
  signal_mask_file = 'sl1_signal_mask.npy'
  noise_mask_file  = 'sl1_noise_mask.npy'

  eval_snr(img_file,signal_mask_file,noise_mask_file)

  img_file         = 'imgs_n1_02.npy'
  signal_mask_file = 'sl2_signal_mask.npy'
  noise_mask_file  = 'sl2_noise_mask.npy'

  eval_snr(img_file,signal_mask_file,noise_mask_file)

  img_file         = 'imgs_n1_03.npy'
  signal_mask_file = 'sl3_signal_mask.npy'
  noise_mask_file  = 'sl3_noise_mask.npy'

  eval_snr(img_file,signal_mask_file,noise_mask_file)


