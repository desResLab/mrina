import matplotlib.pyplot as plt
import numpy as np
fs=8
plt.rc('font',  family='serif')
plt.rc('xtick', labelsize='x-small')
plt.rc('ytick', labelsize='x-small')
plt.rc('text',  usetex=True)
num_pts = 50
n=100
plt.figure(figsize=(4,3))
for noise_percent in [0.01, 0.05, 0.1, 0.25]:
    coeff = np.load('results/corrsqravg' + str(num_pts) + '_noise' + str(noise_percent*100) + '_n'+str(n)+'.npy')
    corravg = np.mean(coeff, axis=1)
    size = len(corravg)
    plt.plot(range(1,size+1), np.abs(corravg))

lgd = ['1\%', '5\%', '10\%', '25\%']
plt.legend(lgd)
plt.xlabel('Distance',fontsize=fs)
plt.ylabel('Correlation Coefficient ($R^2$)',fontsize=fs)
plt.tick_params(labelsize=fs)
plt.savefig('results/diffnoise' + '.png')

noise_percent=0.1
plt.figure(figsize=(4,3))
for n in [50, 100, 250]:
    coeff = np.load('results/corrsqravg' + str(num_pts) + '_noise' + str(noise_percent*100) + '_n'+str(n)+'.npy')
    corravg = np.mean(coeff, axis=1)
    size = len(corravg)
    plt.plot(range(1,size+1), np.abs(corravg))

lgd = ['50','100', '250']
plt.legend(lgd)
plt.xlabel('Distance',fontsize=fs)
plt.ylabel('Correlation Coefficient ($R^2$)',fontsize=fs)
plt.tick_params(labelsize=fs)
plt.savefig('results/diffsamples' + '.png')
