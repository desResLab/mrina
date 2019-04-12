import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.optimize import curve_fit
#from sklearn.metrics import mean_squared_error
home = os.getenv('HOME')
fs=12
plt.rc('font',  family='serif')
plt.rc('xtick', labelsize='x-small')
plt.rc('ytick', labelsize='x-small')
plt.rc('text',  usetex=True)
dir = home + '/Documents/npy75undersamp/'
num_pts = 50
n=100
plt.figure(figsize=(4,3))
for noise_percent in [ 0.05, 0.1, 0.25]:#[0.01, 0.05, 0.1, 0.25]:
    coeff = np.load(dir + 'results/corrsqravg' + str(num_pts) + '_noise' + str(noise_percent*100) + '_n'+str(n)+'.npy')
    corravg = np.mean(coeff, axis=1)
    size = len(corravg)
    plt.plot(range(1,size+1), np.abs(corravg))

lgd = ['5\% noise', '10\% noise', '25\% noise']
plt.legend(lgd)
plt.xlabel('Distance',fontsize=fs)
plt.ylabel('Correlation Coefficient ($R^2$)',fontsize=fs)
plt.tick_params(labelsize=fs)
plt.xticks(np.arange(1, size+1, 50))
plt.tight_layout()
plt.savefig(dir + 'results/diffnoise' + '.png')


def f(x, a, b, n):
    #return a + b*(1-np.exp(-np.exp(n) * x))
    return a * x ** n  / (x ** n + b)
x=np.arange(1,size+1)
y=np.abs(corravg)
popt, pcov = curve_fit(f, x, y, p0=[1800., 20., 1.])
ypred = f(x[1:], *popt)
mse = ((y[1:] - ypred)**2).mean()
print('mse', mse)
#print(y)
#print('mse',mean_squared_error(y,ypred))
plt.figure()
#plt.scatter(x, y)
plt.scatter(x[10:], y[10:])
#plt.plot(x, f(x, *popt), 'r-', label='fit: a=%5.3f, b=%5.3f, n=%5.3f' % tuple(popt))
plt.legend()
plt.show()

noise_percent=0.1
plt.figure(figsize=(4,3))
for n in [50, 100, 250]:
    coeff = np.load(dir + 'results/corrsqravg' + str(num_pts) + '_noise' + str(noise_percent*100) + '_n'+str(n)+'.npy')
    corravg = np.mean(coeff, axis=1)
    size = len(corravg)
    plt.plot(range(1,size+1), np.abs(corravg))

lgd = ['50 samples','100 samples', '250 samples']
plt.legend(lgd)
plt.xlabel('Distance',fontsize=fs)
plt.ylabel('Correlation Coefficient ($R^2$)',fontsize=fs)
plt.tick_params(labelsize=fs)
plt.tight_layout()
plt.savefig(dir + 'results/diffsamples' + '.png')

dir = home + '/Documents/'
noise_percent=0.1
n = 100
plt.figure(figsize=(4,3))
for folder in ['npy5undersamp/', 'npycrceta/', 'npy75undersamp/']:
    coeff = np.load(dir + folder + 'results/corrsqravg' + str(num_pts) + '_noise' + str(noise_percent*100) + '_n'+str(n)+'.npy')
    corravg = np.mean(coeff, axis=1)
    size = len(corravg)
    plt.plot(range(1,size+1), np.abs(corravg))

lgd = ['5\% undersampled','25\% undersampled', '75\% undersampled']
plt.legend(lgd)
plt.xlabel('Distance',fontsize=fs)
plt.ylabel('Correlation Coefficient ($R^2$)',fontsize=fs)
plt.tick_params(labelsize=fs)
plt.tight_layout()
plt.savefig(dir + folder + 'results/diffundersamp' + '.png')
