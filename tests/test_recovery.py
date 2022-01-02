import unittest
import sys,cv2,pywt
import numpy as np
import numpy.linalg as la

from mrina import generateSamplingMask
from mrina import OperatorWaveletToFourier
from mrina import RecoveryL1NormNoisy
from mrina import lsQR
from mrina import OMPRecovery

class recovery_test_suite(unittest.TestCase):

  def setUp(self):        
    # Set parameters
    # Image scale ratio
    size_ratio = 0.1
    # Select an undesampling ratio
    delta = 0.25
    # Wavelet parameters
    waveName = 'haar'
    waveMode = 'zero'

    # Read Image
    self.im = cv2.imread('tests/city.png', cv2.IMREAD_GRAYSCALE)
    # Rescale Image      
    self.im = cv2.resize(self.im,
                         (0,0),
                         fx=size_ratio, 
                         fy=size_ratio, 
                         interpolation=cv2.INTER_NEAREST)

    # Generate an undersampling mask
    omega = generateSamplingMask(self.im.shape, delta, 'bernoulli')
    # Verify the undersampling ratio
    nsamp = np.sum((omega == 1).ravel())/np.prod(omega.shape)
    # Instantiate the operator
    self.A = OperatorWaveletToFourier(self.im.shape, samplingSet=omega[0], waveletName=waveName)
    # Compute the wavelet coefficients
    self.wim = pywt.coeffs_to_array(pywt.wavedec2(self.im, wavelet=waveName, mode=waveMode))[0]
    # Compute the noiseless undersampled measurements
    self.yim = self.A.eval(self.wim, 1)

    # Target SNR
    SNR = 5
    # Signal power. The factor 2 accounts for real/imaginary parts
    yim_pow = la.norm(self.yim.ravel()) ** 2 / (2 * self.yim.size)
    # Noise st. dev.
    self.sigma = np.sqrt(yim_pow / SNR)
    # Noisy measurements
    self.y = self.yim + self.sigma * (np.random.normal(size=self.yim.shape) + 1j * np.random.normal(size=self.yim.shape))
    z_pow = la.norm((self.y - self.yim).ravel()) ** 2 / (2 * self.yim.size)
    # Print message
    print()
    print('----------------------------------------------')
    print('Image size: ', self.im.shape)
    print('Number of complex measurements: ', self.yim.size)
    print('Signal power: ', yim_pow)
    print('Noise power:  ', z_pow)
    print('SNR:          ', yim_pow/z_pow)
    print('----------------------------------------------')
    print()

  def test_noiseless_recovery(self):  

    # Recovery
    wimrec, _ = RecoveryL1NormNoisy(0.01, self.yim, self.A, disp=True, disp_method=False, method='SoS-L1Ball')
    # The recovered coefficients could be complex!
    imrec_cpx = self.A.getImageFromWavelet(wimrec)
    imrec = np.abs(imrec_cpx)
    # Compare true vs reconstructed image
    print('Reconstruction error')
    print('   Absolute ', la.norm((self.im - imrec_cpx).ravel()))
    print('   Relative ', la.norm((self.im - imrec_cpx).ravel())/la.norm(self.im.ravel()))

  def test_cs_recovery(self):     

    # Parameter eta
    eta = np.sqrt(2 * self.y.size) * self.sigma
    # Recovery
    wimrec_noisy_cpx, _ = RecoveryL1NormNoisy(eta, self.y, self.A, disp=True, disp_method=False, method='BPDN')
    # The recovered coefficients could be complex!
    imrec_noisy_cpx = self.A.getImageFromWavelet(wimrec_noisy_cpx)
    imrec_noisy = np.abs(imrec_noisy_cpx)
    # Compare true vs reconstructed image
    print('Reconstruction error')
    print('   Absolute ', la.norm((self.im - imrec_noisy_cpx).ravel()))
    print('   Relative ', la.norm((self.im - imrec_noisy_cpx).ravel())/la.norm(self.im.ravel()))

    # Flag to use LSQR solver 
    useLSQR = True
    # Support of noisy solution
    wim_supp = np.where(np.abs(wimrec_noisy_cpx) > 1E-4 * la.norm(wimrec_noisy_cpx.ravel(), np.inf), True, False)
    # Restriction of the operator
    Adeb = self.A.colRestrict(wim_supp)
    # Solve least-squares problem
    if(useLSQR):
        lsqr = lsQR(Adeb)  
        lsqr.solve(self.y[Adeb.samplingSet])
        wimrec_noisy_cpx_deb = np.zeros(Adeb.wavShape,dtype=np.complex)
        wimrec_noisy_cpx_deb[Adeb.basisSet] = lsqr.x[:]
    else:
        from mrina import MinimizeSumOfSquares
        wimrec_noisy_cpx_deb, _ = MinimizeSumOfSquares(y, Adeb, disp=True, printEvery=100)
    # The recovered coefficients could be complex!
    imrec_noisy_cpx_deb = Adeb.getImageFromWavelet(wimrec_noisy_cpx_deb)
    imrec_noisy_deb = np.abs(imrec_noisy_cpx_deb)
    # Compare true vs reconstructed image
    print('Reconstruction error - no debiasing')
    print('   Absolute ', la.norm((self.im - imrec_noisy_cpx).ravel()))
    print('   Relative ', la.norm((self.im - imrec_noisy_cpx).ravel())/la.norm(self.im.ravel()))
    print('Reconstruction error - debiasing')
    print('   Absolute ', la.norm((self.im - imrec_noisy_cpx_deb).ravel()))
    print('   Relative ', la.norm((self.im - imrec_noisy_cpx_deb).ravel())/la.norm(self.im.ravel()))

  def test_omp_recovery(self):   

    # Recovery
    wimrec_noisy_cpx, _ = OMPRecovery(self.A, self.y)
    # The recovered coefficients could be complex!
    imrec_noisy_cpx = self.A.getImageFromWavelet(wimrec_noisy_cpx)
    imrec_noisy = np.abs(imrec_noisy_cpx)
    # Compare true vs reconstructed image
    print('OMP Reconstruction error')
    print('   Absolute ', la.norm((self.im - imrec_noisy_cpx).ravel()))
    print('   Relative ', la.norm((self.im - imrec_noisy_cpx).ravel())/la.norm(self.im.ravel()))

if __name__ == '__main__':
    unittest.main()
