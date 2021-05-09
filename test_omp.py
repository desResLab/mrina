import cv2
import numpy as np
import pywt
import matplotlib.pyplot as plt
# Import from suite
from maps import OperatorLinear, OperatorWaveletToFourier
from solver_omp import lsQR,OMPRecovery
from CSRecoverySuite import generateSamplingMask
import time

def linearTest():
  print('')
  print('--- TEST FOR LINEAR COMPLEX SYSTEM')
  print('')
  # Solve the SQD system
  #  [ 2  1 ] [x] = [2]
  #  [ 1 -3 ] [y]   [0]

  m = 100 # Number of Rows
  n = 10  # Number of Columns
  aMat = np.random.randn(m,n) + 1j * np.random.randn(m,n)
  sol  = np.random.randn(n) + 1j * np.random.randn(n)
  rhs  = np.dot(aMat,sol)
  
  # Initialize the operator
  A = OperatorLinear(aMat)
  
  # Init lsQR
  lsqr = lsQR(A)
  
  # Solve the least-square problem with LSQR
  lsqr.solve(rhs, itnlim=100, show=True)

  # Try with the numpy least squares solver
  lsSol = np.linalg.lstsq(aMat, rhs, rcond=None)[0]

  # Plot the Residual
  print('Residual Norm with True Solution: ', np.linalg.norm(np.dot(aMat,sol)-rhs))
  print('Residual Norm LSQR: ', np.linalg.norm(np.dot(aMat,lsqr.x)-rhs))
  print('Residual Norm Numpy LS: ', np.linalg.norm(np.dot(aMat,lsSol)-rhs))
  #for loopA in range(n):
  #  print(sol[loopA], lsqr.x[loopA], lsSol[loopA])

def ompTest():
  print('')
  print('--- TEST FOR OMP')
  print('')

  m = 50
  n = 200
  p = 10

  np.random.seed(1345)
  aMat = np.random.randn(m,n) + 1j * np.random.randn(m,n)
  x = np.zeros(n,dtype=np.complex)
  index_set = np.random.randint(0, n, size=p)
  x[index_set] = np.random.normal(loc = 6, scale = 1, size = p) + 1j * np.random.normal(loc = 6, scale = 1, size = p)
  b = np.dot(aMat,x)

  # Initialize the operator
  A = OperatorLinear(aMat)

  # Solve with OMP
  ompSol = OMPRecovery(A, b, tol=1E-12, progressInt=1, ompMethod='omp')[0]

  # Print the original and reconstructed solution
  print()
  print("{:<6s} {:<15s} {:<15s}".format('Index','True Sol.','Recovered Sol.'))
  for loopA in range(n):
    if(ompSol[loopA] > 0.0):
      print('{:<6d} {:<15.3f} {:<15.3f}'.format(loopA,x[loopA],ompSol[loopA]))

def imageTest():

  print('')
  print('--- OMP RECONSTRUCTION ON IMAGE')
  print('')

  # Set parameters
  wType = 'haar'  
  waveMode = 'zero'
  size_ratio = 0.5
  uratio = 0.75

  # Open Image
  im = cv2.imread('verification/city.png', cv2.IMREAD_GRAYSCALE)

  # Rescale Image
  im = cv2.resize(im,    # original image
                  (0,0), # set fx and fy, not the final size
                  fx=size_ratio, 
                  fy=size_ratio, 
                  interpolation=cv2.INTER_NEAREST)

  # Compute Wavelet Coefficient
  wim = pywt.coeffs_to_array(pywt.wavedec2(im, wavelet=wType, mode=waveMode))[0]

  # Generate undersampling mask
  omega = generateSamplingMask(im.shape, uratio, 'vardengauss')

  # Create WtF operator
  A = OperatorWaveletToFourier(im.shape, samplingSet=omega[0], waveletName=wType)

  # Generate undersampled measurements
  yim = A.eval(wim, 1)

  print('[SINGLE IMAGE - RANDOM UNDERSAMPLING - L1-NORM RECOVERY]')
  print('    Input shape:    {:d} x {:d}'.format(A.inShape[0], A.inShape[1]))
  print('    Output shape:   {:d} x {:d}'.format(A.outShape[0], 1))
  print('    Operator norm:  {:1.3e}'.format(A.norm()))

  # Recover Image
  wimrec_noisy_cpx, _ = OMPRecovery(A, yim)
  imrec_noisy_cpx = A.getImageFromWavelet(wimrec_noisy_cpx)
  imrec_noisy = np.abs(imrec_noisy_cpx)

  # Compare true vs reconstructed image
  print('Reconstruction error')
  print('   Absolute ', np.linalg.norm((im - imrec_noisy_cpx).ravel()))
  print('   Relative ', np.linalg.norm((im - imrec_noisy_cpx).ravel())/np.linalg.norm(im.ravel()))

  # Plot Comparison
  plt.figure(figsize=(6,3))
  plt.subplot(1,2,1)
  plt.imshow(im, cmap='gray')
  plt.title('true')
  plt.axis('off')
  plt.subplot(1,2,2)
  plt.imshow(imrec_noisy, cmap='gray')
  plt.title('OMP recovery')
  plt.axis('off')
  plt.show()

# MAIN
if __name__ == '__main__':

  start_time = time.time()
  # Perform Simple Linear Test
  linearTest()
  # Perform Test for OMP with linear operators
  ompTest()
  # Perform Test on an image reconstruction using a Fourier-Wavelet operator
  imageTest()
  print()
  print("Total time: %s s" % (time.time() - start_time))
