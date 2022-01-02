import unittest
import numpy as np
import numpy.linalg as npla
import numpy.fft as fft
import matplotlib.pyplot as plt
import scipy.optimize
import os

class l2_test_suite(unittest.TestCase):

  def setUp(self):

    # Load image
    flow_data = np.load('paper/03_idealaorta/imgs_n1.npy')
    self.imshape = (flow_data.shape[3], flow_data.shape[4])
    print('shape of flow data:  ', flow_data.shape)
    # Get vmax
    vmax = np.max(np.abs(flow_data[0, 1::, 0, :, :]))
    print('vmax:                ', vmax)
    # Set venc
    venc = 1.0 * vmax
    print('venc:                ', venc)
    print('venc (factor):       ', venc/vmax)
    # Get complex images
    Xo = flow_data[0, 0, 0, :, :]
    Xi = flow_data[0, 0, 0, :, :] * np.exp(1j * np.pi * flow_data[0, 1, 0, :, :] / venc)
    Xj = flow_data[0, 0, 0, :, :] * np.exp(1j * np.pi * flow_data[0, 2, 0, :, :] / venc)
    Xk = flow_data[0, 0, 0, :, :] * np.exp(1j * np.pi * flow_data[0, 3, 0, :, :] / venc)
    # Add images to list
    self.Xim = [ Xo, Xi, Xj, Xk ]
    # Sampling fraction
    self.delta = 0.25


  def test_l2_rec(self):

    # ### Bernoulli mask

    # Sampling set
    #   Bernoulli does not need fftshift
    S = np.where(np.random.uniform(size=self.imshape) <= self.delta, True, False)

    # Expected value of the reconstructed images over the noise conditioned on the mask

    EXim = []
    for I in range(4):
        EXim.append(fft.ifft2(S * fft.fft2(self.Xim[I], norm='ortho'), norm='ortho'))

    # Covariance kernel over the noise conditioned on the mask

    K = fft.fftshift(fft.ifft2(S)) / self.delta

    # Expectation over the mask and noise

    EXim = []
    for I in range(4):
        EXim.append(fft.ifft2(self.delta * fft.fft2(self.Xim[I], norm='ortho'), norm='ortho'))

    # Covariance kernel over the noise and the mask

    KXim = []
    for I in range(4):
        W = self.delta * (1 - self.delta) * np.abs(fft.fft2(self.Xim[I], norm='ortho')) ** 2 / np.sqrt(np.prod(self.imshape))
        KXim.append(fft.ifft2(W * fft.fft2(self.Xim[I], norm='ortho'), norm='ortho'))

    # ### Gaussian mask

    # Sampling set
    kx = np.linspace(0, self.imshape[0] - 1, self.imshape[0])
    kx = fft.fftshift(np.where(kx < self.imshape[0] // 2, kx, kx - self.imshape[0]))
    ky = np.linspace(0, self.imshape[1] - 1, self.imshape[1])
    ky = fft.fftshift(np.where(ky < self.imshape[1] // 2, ky, ky - self.imshape[1]))
    [ kx, ky ] = np.meshgrid(kx, ky)
    f = lambda t : self.delta - np.sum(np.exp(-t * (kx ** 2 + ky ** 2)).ravel()) / (self.imshape[0] * self.imshape[1])
    tsamp, rnfo = scipy.optimize.toms748(f, 1E-6, 1E+6, xtol=1E-6, full_output=True, disp=True)

    W = np.exp(-tsamp * (kx ** 2 + ky ** 2))
    S = np.where(np.random.uniform(size=self.imshape) <= W, True, False)
    S = fft.fftshift(S) 

    # Expected value of the reconstructed images over the noise conditioned on the mask

    EXim = []
    for I in range(4):
        EXim.append(fft.ifft2(S * fft.fft2(self.Xim[I], norm='ortho'), norm='ortho'))

    # Covariance kernel over the noise conditioned on the mask

    K = fft.fftshift(fft.ifft2(S))
    K = K / np.abs(K).max()

    # Expectation over the mask and noise

    EXim = []
    for I in range(4):
        EXim.append(fft.ifft2(fft.fftshift(W) * fft.fft2(self.Xim[I], norm='ortho'), norm='ortho'))

    # Covariance kernel over the noise and the mask

    KXim = []

    for I in range(4):
        _W = fft.fftshift(W * (1 - W)) * np.abs(fft.fft2(self.Xim[I], norm='ortho')) ** 2 / np.sqrt(np.prod(self.imshape))
        KXim.append(fft.ifft2(_W * fft.fft2(self.Xim[I], norm='ortho'), norm='ortho'))


if __name__ == '__main__':
    unittest.main()
