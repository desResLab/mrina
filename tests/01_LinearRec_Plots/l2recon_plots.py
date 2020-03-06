# %%
# Import
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as sciopt
import scipy
from CSRecoverySuite import generateSamplingMask
from numpy.random import RandomState
import matplotlib.pyplot as plt

# %%
# Numerical experiments for l2-recovery
#   This script reproduces the figures for Section X.Y on the paper.
#   The source image is the same for all experiments.
im = plt.imread('tests/01_LinearRec_Plots/nd_small.jpg')
im = np.mean(im, axis=2)
im = im / np.max(im.ravel())
imsz = im.shape

#   The random seed is fixed to zero. As long as the behavior of the random
#   number generator does not change, the results should be the same.
np.random.seed(seed=0)
np.random.RandomState(seed=0)
savefig = True

# *** Minimum l2-norm recovery
# %%
# - Experiment #1:
#       Compare the convolution kernel associated to the covariance matrix for
#       sets sampled from different distributions
#       Options = bernoulli, vardentri, vardengauss, vardenexp, halton
# Sampling fraction
delta = 0.1
# Sampling masks
Sber = 1 - generateSamplingMask(imsz, 1 - delta, saType='bernoulli', num_patterns=1)[0, :, :]
Sgau = 1 - generateSamplingMask(imsz, 1 - delta, saType='vardengauss', num_patterns=1)[0, :, :]
print(' Sampled...')
print('   {:d}/{:d} coefficients with Bernoulli mask'.format(np.sum(Sber.ravel()), im.size))
print('   {:d}/{:d} coefficients with Gaussian mask'.format(np.sum(Sgau.ravel()), im.size))
# Kernel for conditional covariance
Kber = np.fft.ifft2(Sber)
Kber = Kber / np.max(np.abs(Kber.ravel()))
Kgau = np.fft.ifft2(Sgau)
Kgau = Kgau / np.max(np.abs(Kgau.ravel()))
# Kernel for full covariance
mu_ber = delta * np.ones(im.shape)
#   Compute explicitly \mu for Gaussian sampling
c = 2 * np.sqrt(delta / np.pi)
s, rnfo = sciopt.toms748(lambda t: scipy.special.erf(t) - c * t, 1E-9, 2/c, xtol=1E-3, full_output=True, disp=True)
s = 1 / (s * np.sqrt(2))
x = np.linspace(-1, 1, num=im.shape[1])
y = np.linspace(-1, 1, num=im.shape[0])
mu_gau = np.zeros(im.shape)
for Ii in range(im.shape[0]):
  for Ij in range(im.shape[1]):
    mu_gau[Ii, Ij] = np.exp( -(x[Ij]**2 + y[Ii]**2) / (2 * s**2) )
mu_gau = np.fft.fftshift(mu_gau)
#   Kernels
Wber = np.fft.ifft2(mu_ber * (1 - mu_ber) * (np.abs(np.fft.fft2(im, norm='ortho')) ** 2), norm='ortho')
Wgau = np.fft.ifft2(mu_gau * (1 - mu_gau) * (np.abs(np.fft.fft2(im, norm='ortho')) ** 2), norm='ortho')
print(' Norms...')
print('   |Wber|_inf = {:5.3f}'.format(np.max(np.abs(Wber.ravel()))))
print('   |Wgau|_inf = {:5.3f}'.format(np.max(np.abs(Wgau.ravel()))))
# %%
#   Plot masks
cmap = 'gray'
cmapk = 'seismic'
fig, ax = plt.subplots()
imax = ax.imshow(np.fft.fftshift(Sber), aspect='equal', vmin=0, vmax=1, cmap=cmap)
ax.set(xlabel=None, ylabel=None, title='$\Omega$\n{:2.0f}% sampled'.format(100 * delta))
ax.set_axis_off()
fig.colorbar(imax)
if( savefig ):
  plt.savefig(fname='tests/01_LinearRec_Plots/fig/sampling_mask_bernoulli_{:2.0f}p.png'.format(100 * delta), dpi=600, bbox_inches='tight')

fig, ax = plt.subplots()
imax = ax.imshow(np.fft.fftshift(Sgau), aspect='equal', vmin=0, vmax=1, cmap=cmap)
ax.set(xlabel=None, ylabel=None, title='$\Omega$\n{:2.0f}% sampled'.format(100 * delta))
ax.set_axis_off()
fig.colorbar(imax)
if( savefig ):
  plt.savefig(fname='tests/01_LinearRec_Plots/fig/sampling_mask_gaussian_{:2.0f}p.png'.format(100 * delta), dpi=600, bbox_inches='tight')

fig, ax = plt.subplots()
imax = ax.imshow(np.abs(np.fft.fftshift(Kber)), aspect='equal', vmin=0, vmax=1, cmap=cmap)
ax.set(xlabel=None, ylabel=None, title='$K_\Omega$\n{:2.0f}% sampled'.format(100 * delta))
ax.set_axis_off()
fig.colorbar(imax)
if( savefig ):
  plt.savefig(fname='tests/01_LinearRec_Plots/fig/KO_mask_bernoulli_{:2.0f}p.png'.format(100 * delta), dpi=600, bbox_inches='tight')

fig, ax = plt.subplots()
imax = ax.imshow(np.abs(np.fft.fftshift(Kgau)), aspect='equal', vmin=0, vmax=1, cmap=cmap)
ax.set(xlabel=None, ylabel=None, title='$K_\Omega$\n{:2.0f}% sampled'.format(100 * delta))
ax.set_axis_off()
fig.colorbar(imax)
if( savefig ):
  plt.savefig(fname='tests/01_LinearRec_Plots/fig/KO_mask_gaussian_{:2.0f}p.png'.format(100 * delta), dpi=600, bbox_inches='tight')

fig, ax = plt.subplots()
imax = ax.imshow(np.fft.fftshift(mu_ber), aspect='equal', vmin=0, vmax=1, cmap=cmap)
ax.set(xlabel=None, ylabel=None, title='$\mu$ for Bernoulli sampling\n{:2.0f}% sampled'.format(100 * delta))
ax.set_axis_off()
fig.colorbar(imax)
if( savefig ):
  plt.savefig(fname='tests/01_LinearRec_Plots/fig/mu_mask_bernoulli_{:2.0f}p.png'.format(100 * delta), dpi=600, bbox_inches='tight')

fig, ax = plt.subplots()
imax = ax.imshow(np.fft.fftshift(mu_gau), aspect='equal', vmin=0, vmax=1, cmap=cmap)
ax.set(xlabel=None, ylabel=None, title='$\mu$ for Gaussian sampling\n{:2.0f}% sampled'.format(100 * delta))
ax.set_axis_off()
fig.colorbar(imax)
if( savefig ):
  plt.savefig(fname='tests/01_LinearRec_Plots/fig/mu_mask_gaussian_{:2.0f}p.png'.format(100 * delta), dpi=600, bbox_inches='tight')

fig, ax = plt.subplots()
imax = ax.imshow(np.abs(np.fft.fftshift(Wber)), aspect='equal', vmin=0, vmax=5.1, cmap=cmap)
ax.set(xlabel=None, ylabel=None, title='$K_{{\mu, x}}$\n{:2.0f}% sampled'.format(100 * delta))
ax.set_axis_off()
fig.colorbar(imax)
if( savefig ):
  plt.savefig(fname='tests/01_LinearRec_Plots/fig/Kmux_mask_bernoulli_{:2.0f}p.png'.format(100 * delta), dpi=600, bbox_inches='tight')

fig, ax = plt.subplots()
imax = ax.imshow(np.abs(np.fft.fftshift(Wgau)), aspect='equal', vmin=0, vmax=0.5, cmap=cmap)
ax.set(xlabel=None, ylabel=None, title='$K_{{\mu, x}}$\n{:2.0f}% sampled'.format(100 * delta))
ax.set_axis_off()
fig.colorbar(imax)
if( savefig ):
  plt.savefig(fname='tests/01_LinearRec_Plots/fig/Kmux_mask_gaussian_{:2.0f}p.png'.format(100 * delta), dpi=600, bbox_inches='tight')

# %%
# - Experiment #2:
#       See the effects of shrinkage for the full covariance via sampling
#       for Bernoulli and Gaussian masks
# Number of MonteCarlo samples
ns = 1000
eta_frac = [ 0.25, 0.5, 0.75 ]
Xber = 1j * np.zeros((ns,) + im.shape)
Xgau = 1j * np.zeros((ns,) + im.shape)
# Sampling fraction
delta = 0.1
# Expected signal power
xpow_ber = np.sum(mu_ber * (np.abs(np.fft.fft2(im, norm='ortho')) ** 2))
xpow_gau = np.sum(mu_gau * (np.abs(np.fft.fft2(im, norm='ortho')) ** 2))
print('Average signal power...')
print('   Bernoulli mask: {:5.3f}'.format(xpow_ber))
print('   Gaussian mask:  {:5.3f}'.format(xpow_gau))
for Ie in range(3):
  # eta
  eta_ber = eta_frac[Ie] * np.sqrt(xpow_ber)
  eta_gau = eta_frac[Ie] * np.sqrt(xpow_gau)
  # Shrinkage function
  gamma_ber = lambda y : np.maximum(0.0, 1 - eta_ber / np.linalg.norm(y.ravel()))
  gamma_gau = lambda y : np.maximum(0.0, 1 - eta_ber / np.linalg.norm(y.ravel()))
  print('Sampling...')
  for Is in range(ns):
    # Sampling sets
    Sber = 1 - generateSamplingMask(imsz, 1 - delta, saType='bernoulli', num_patterns=1)[0, :, :]
    Sgau = 1 - generateSamplingMask(imsz, 1 - delta, saType='vardengauss', num_patterns=1)[0, :, :]
    # Sampling
    y_ber = Sber * np.fft.fft2(im, norm='ortho')
    y_gau = Sgau * np.fft.fft2(im, norm='ortho')
    # Recovery
    x_ber = gamma_ber(y_ber) * np.fft.ifft2(y_ber, norm='ortho')
    x_gau = gamma_gau(y_gau) * np.fft.ifft2(y_gau, norm='ortho')
    # Sample
    Xber[Is, :, :] = x_ber
    Xgau[Is, :, :] = x_gau
    print(' [{:d}] Experiment {:d} '.format(Ie, Is))

  np.save('tests/01_LinearRec_Plots/npy/samples_bernoulli_{:2.0f}s_eta_{:2.0f}.npy'.format(100 * delta, 100 * eta_frac[Ie]), Xber)
  np.save('tests/01_LinearRec_Plots/npy/samples_gaussian_{:2.0f}s_eta_{:2.0f}.npy'.format(100 * delta, 100 * eta_frac[Ie]), Xgau)

# %%
# Compute bins
delta = 0.1

# Bernoulli sampling
Xber = np.load('tests/01_LinearRec_Plots/npy/samples_bernoulli_10s_eta_25.npy')
Xber_avg = np.mean(Xber, axis=0)
ber_chi_25 = np.linalg.norm(Xber - Xber_avg, axis=(1, 2)) ** 2
ber_bins_25 = np.linspace(np.min(ber_chi_25), np.max(ber_chi_25), 100)

Xber = np.load('tests/01_LinearRec_Plots/npy/samples_bernoulli_10s_eta_50.npy')
Xber_avg = np.mean(Xber, axis=0)
ber_chi_50 = np.linalg.norm(Xber - Xber_avg, axis=(1, 2)) ** 2
ber_bins_50 = np.linspace(np.min(ber_chi_50), np.max(ber_chi_50), 100)

Xber = np.load('tests/01_LinearRec_Plots/npy/samples_bernoulli_10s_eta_75.npy')
Xber_avg = np.mean(Xber, axis=0)
ber_chi_75 = np.linalg.norm(Xber - Xber_avg, axis=(1, 2)) ** 2
ber_bins_75 = np.linspace(np.min(ber_chi_75), np.max(ber_chi_75), 100)

# Gaussian sampling
Xgau = np.load('tests/01_LinearRec_Plots/npy/samples_gaussian_10s_eta_25.npy')
Xgau_avg = np.mean(Xgau, axis=0)
gau_chi_25 = np.linalg.norm(Xgau - Xgau_avg, axis=(1, 2)) ** 2
gau_bins_25 = np.linspace(np.min(gau_chi_25), np.max(gau_chi_25), 100)

Xgau = np.load('tests/01_LinearRec_Plots/npy/samples_gaussian_10s_eta_50.npy')
Xgau_avg = np.mean(Xgau, axis=0)
gau_chi_50 = np.linalg.norm(Xgau - Xgau_avg, axis=(1, 2)) ** 2
gau_bins_50 = np.linspace(np.min(gau_chi_50), np.max(gau_chi_50), 100)

Xgau = np.load('tests/01_LinearRec_Plots/npy/samples_gaussian_10s_eta_75.npy')
Xgau_avg = np.mean(Xgau, axis=0)
gau_chi_75 = np.linalg.norm(Xgau - Xgau_avg, axis=(1, 2)) ** 2
gau_bins_75 = np.linspace(np.min(gau_chi_75), np.max(gau_chi_75), 100)

# %%
plt.hist(ber_chi_25, ber_bins_25, alpha=0.75, label='$\eta = 0.25 \times E_{{\Omega}}(\|F_{{\Omega}}x\|_2^2)^{{1/2}}$')
plt.hist(ber_chi_50, ber_bins_50, alpha=0.75, label='$\eta = 0.50 \times E_{{\Omega}}(\|F_{{\Omega}}x\|_2^2)^{{1/2}}$')
plt.hist(ber_chi_75, ber_bins_75, alpha=0.75, label='$\eta = 0.75 \times E_{{\Omega}}(\|F_{{\Omega}}x\|_2^2)^{{1/2}}$')
plt.legend(loc='upper right')
plt.xlim((0, 10000))
plt.ylim((0, 10000))
plt.yscale('symlog')
plt.title('Bernoulli sampling\n{:2.0f}% sampled'.format(100 * delta))
if( savefig ):
  plt.savefig(fname='tests/01_LinearRec_Plots/fig/chisq_stat_bernoulli_{:2.0f}p.png'.format(100 * delta), dpi=600, bbox_inches='tight')
plt.show()

plt.hist(gau_chi_25, gau_bins_25, alpha=0.75, label='25%')
plt.hist(gau_chi_50, gau_bins_50, alpha=0.75, label='50%')
plt.hist(gau_chi_75, gau_bins_75, alpha=0.75, label='75%')
plt.legend(loc='upper right')
plt.xlim((0, 10000))
plt.ylim((0, 10000))
plt.yscale('symlog')
plt.title('Gaussian sampling\n{:2.0f}% sampled'.format(100 * delta))
if( savefig ):
  plt.savefig(fname='tests/01_LinearRec_Plots/fig/chisq_stat_gaussian_{:2.0f}p.png'.format(100 * delta), dpi=600, bbox_inches='tight')
plt.show()
