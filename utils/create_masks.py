import numpy as np
import matplotlib.pyplot as plt
import cv2

# ====
# MAIN
# ====
if __name__ == "__main__":  

  """
  Creates a boolean mack from a png black & white image
  """

  # name = 'sl3_noise.png'
  name = 'sl3_signal.png'

  im = cv2.imread(name, cv2.IMREAD_GRAYSCALE)
  im_bin = (im > 10)
  np.save(name+'_mask.npy',im_bin)







