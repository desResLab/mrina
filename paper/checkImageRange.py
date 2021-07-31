import numpy as np
import matplotlib.pyplot as plt

# MAIN 
if __name__ == '__main__':

  # Get Image Ranges
  img = np.load('./02_ndimg/imgs_n1.npy')
  print('NDImg Shape: ',img.shape)
  for loopA in range(img.shape[1]):
    print('NDImg %d, min: %f, max: %f' % (loopA,img[0,loopA].min(),img[0,loopA].max()))

  img = np.load('./03_poiseuilleaxis1/imgs_n1.npy')
  print('Poiseuille 1 Shape: ',img.shape)
  for loopA in range(img.shape[1]):
    print('Poiseuille 1 %d, min: %f, max: %f' % (loopA,img[0,loopA].min(),img[0,loopA].max()))

  img = np.load('./04_poiseuilleaxis2/imgs_n1.npy')
  print('Poiseuille 2 Shape: ',img.shape)
  for loopA in range(img.shape[1]):
    print('Poiseuille 2 %d, min: %f, max: %f' % (loopA,img[0,loopA].min(),img[0,loopA].max()))

  img = np.load('./05_idealaorta/imgs_n1.npy')
  print('Aorta Ideal Shape: ',img.shape)
  for loopA in range(img.shape[1]):
    print('Aorta Ideal %d, min: %f, max: %f' % (loopA,img[0,loopA].min(),img[0,loopA].max()))

  img = np.load('./06_aortamri/imgs_n1.npy')
  print('Aorta MRI Shape: ',img.shape)
  for loopA in range(img.shape[1]):
    print('Aorta MRI %d, min: %f, max: %f' % (loopA,img[0,loopA].min(),img[0,loopA].max()))
