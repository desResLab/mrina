import matplotlib.pyplot as plt
import numpy as np
import os
import math
from scipy.optimize import curve_fit
#from sklearn.metrics import mean_squared_error
home = os.getenv('HOME')
fs=12
plt.rc('font',  family='serif')
plt.rc('xtick', labelsize='x-small')
plt.rc('ytick', labelsize='x-small')
plt.rc('text',  usetex=True)
dir = home + '/Documents/undersampled/poiseuille/debiasing/'
num_pts = 50
n=100
p=0.75
start = 0
end = 30 #size
interval = int((end-start)/4)#int(math.ceil(((end-start)/4) / 10.0)) * 10))
type='bernoulli'
def find_max():
    mx=0.
    p=0.75
    for v in range(0,4):
        plt.figure(figsize=(4,3))
        for noise_percent in [0.01,0.05,0.1,0.3]:
            coeff = np.load(dir + 'results/corrsqravg' + str(num_pts) + '_noise' + str(int(noise_percent*100)) + '_p' + str(int(p*100)) + type +'_n'+str(n) + '.npy')
            coeff = coeff[v]
            corravg = np.mean(coeff, axis=1)
            corrmax = np.percentile(coeff, 90, axis=1)
            print(np.amax(corrmax))
            mx = max(mx, np.amax(corrmax))

    noise_percent=0.1
    for v in range(0,4):
        plt.figure(figsize=(4,3))
        for p in [0.25,0.5,0.75]:#[0.25, 0.5, 0.75]:
            coeff = np.load(dir + 'results/corrsqravg' + str(num_pts) + '_noise' + str(int(noise_percent*100)) + '_p' + str(int(p*100)) + type +'_n'+str(n) + '.npy')
            coeff = coeff[v]
            corravg = np.mean(coeff, axis=1)
            corrmax = np.percentile(coeff, 90, axis=1)
            mx = max(mx, np.max(corrmax))

    # noise_percent=0.1
    # p=0.5
    # for v in range(0,4):
    #     plt.figure(figsize=(4,3))
    #     for samptype in ['bernoulli', 'poisson', 'halton']:
    #         coeff = np.load(dir + 'results/corrsqravg' + str(num_pts) + '_noise' + str(int(noise_percent*100)) + '_p' + str(int(p*100)) + type +'_n'+str(n) + '.npy')
    #         coeff = coeff[v]
    #         corravg = np.mean(coeff, axis=1)
    #         corrmax = np.percentile(coeff, 90, axis=1)
    #         mx = max(mx, np.max(corrmax))
    return mx

max = find_max()
for v in range(0,4):
    plt.figure(figsize=(4,3))
    for noise_percent in [0.01,0.05,0.1,0.3]:
        coeff = np.load(dir + 'results/corrsqravg' + str(num_pts) + '_noise' + str(int(noise_percent*100)) + '_p' + str(int(p*100)) + type +'_n'+str(n) + '.npy')
        coeff = coeff[v]
        corravg = np.mean(coeff, axis=1)
        corrmin = np.percentile(coeff, 10, axis=1)
        corrmax = np.percentile(coeff, 90, axis=1)
        size = len(corravg)
        #plt.plot(range(1,size+1), np.abs(corravg))
        #plt.fill_between(range(1,size+1), corrmin, corrmax)
        plt.plot(range(start+1,end+1), np.abs(corravg)[start:end])
        plt.fill_between(range(start+1,end+1), corrmin[start:end], corrmax[start:end], alpha=0.2)
    lgd = ['1\% noise', '5\% noise', '10\% noise', '30\% noise']
    plt.legend(lgd)
    plt.xlabel('Distance',fontsize=fs)
    plt.ylabel('Correlation Coefficient ($R^2$)',fontsize=fs)
    plt.tick_params(labelsize=fs)
    #plt.xticks(np.arange(start, end+1, 50))
    plt.ylim(top=max)
    plt.xticks(np.arange(start+1, end+1,interval))
    plt.tight_layout()
    plt.savefig(dir + 'results/diffnoise' + str(start) + 'to' + str(end) + '_p' + str(int(p*100)) + '_v' + str(v) + '.png')
    #plt.draw()
#plt.show()

noise_percent=0.1
for v in range(0,4):
    plt.figure(figsize=(4,3))
    for p in [0.25,0.5, 0.75]:
        coeff = np.load(dir + 'results/corrsqravg' + str(num_pts) + '_noise' + str(int(noise_percent*100)) + '_p' + str(int(p*100)) + type +'_n'+str(n) + '.npy')
        coeff = coeff[v]
        corravg = np.mean(coeff, axis=1)
        corrmin = np.percentile(coeff, 10, axis=1)
        corrmax = np.percentile(coeff, 90, axis=1)
        size = len(corravg)
        plt.plot(range(start+1,end+1), np.abs(corravg)[start:end])
        plt.fill_between(range(start+1,end+1), corrmin[start:end], corrmax[start:end], alpha=0.2)
    lgd = ['25\% undersampling', '50\% undersampling', '75\% undersampling']
    plt.legend(lgd)
    plt.xlabel('Distance',fontsize=fs)
    plt.ylabel('Correlation Coefficient ($R^2$)',fontsize=fs)
    plt.tick_params(labelsize=fs)
    #plt.xticks(np.arange(start, end+1, 50))
    plt.ylim(top=max)
    plt.xticks(np.arange(start+1, end+1,interval))
    plt.tight_layout()
    plt.savefig(dir + 'results/diffundersamp' + str(start) + 'to' + str(end) + '_noise' + str(int(noise_percent*100)) + '_v' + str(v) + '.png')
    #plt.draw()
#plt.show()

noise_percent=0.05
p=0.75
for v in range(0,4):
    plt.figure(figsize=(4,3))
    for samptype in ['bernoulli', 'bpoisson', 'halton', 'vardengauss','vardentri', 'vardenexp']:
        coeff = np.load(dir + 'results/corrsqravg' + str(num_pts) + '_noise' + str(int(noise_percent*100)) + '_p' + str(int(p*100)) + samptype +'_n'+str(n) + '.npy')
        coeff = coeff[v]
        corravg = np.mean(coeff, axis=1)
        corrmin = np.percentile(coeff, 10, axis=1)
        corrmax = np.percentile(coeff, 90, axis=1)
        size = len(corravg)
        plt.plot(range(start+1,end+1), np.abs(corravg)[start:end])
        plt.fill_between(range(start+1,end+1), corrmin[start:end], corrmax[start:end], alpha=0.2)
    lgd = ['Bernoulli undersampling','Poisson undersampling', 'Halton undersampling', 'Gauss density undersampling', 'Tri density undersampling','Exp density undersampling']
    plt.legend(lgd)
    plt.xlabel('Distance',fontsize=fs)
    plt.ylabel('Correlation Coefficient ($R^2$)',fontsize=fs)
    plt.tick_params(labelsize=fs)
    plt.ylim(top=max)
    #plt.xticks(np.arange(start, end+1, 50))
    plt.xticks(np.arange(start+1, end+1,interval))
    plt.tight_layout()
    plt.savefig(dir + 'results/diffsamptype' + str(start) + 'to' + str(end) + '_noise' + str(int(noise_percent*100)) + '_p' + str(int(p*100)) + '_v' + str(v) + '.png')
#     #plt.draw()
# plt.show()
