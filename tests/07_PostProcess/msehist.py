import matplotlib.pyplot as plt
sys.path.append('../../')
from genSamples import linear_reconstruction, recover_vel #threshold for zero mag
from genSamples import getKspace
import numpy as np
import sys
import os
from matplotlib.ticker import PercentFormatter, ScalarFormatter
from CSRecoverySuite import CSRecovery,CSRecoveryDebiasing, Operator4dFlow, pywt2array, array2pywt, crop
home = os.getenv('HOME')
fs=12
plt.rc('font',  family='serif')
plt.rc('xtick', labelsize='x-small')
plt.rc('ytick', labelsize='x-small')
plt.rc('text',  usetex=True)
num_samples = 100
def get_files(dir, recdir, noise_percent, p, type):
    patterndir = home + "/apps/undersampled/poiseuille/npy/"
    fourier_file = dir + 'noisy_noise' + str(int(noise_percent*100)) + '_n' + str(num_samples) + '.npy'
    undersample_file = patterndir + 'undersamplpattern_p' + str(int(p*100)) + type +  '_n' + str(num_samples) + '.npy'
    pattern = np.load(undersample_file)
    omega = pattern
    if len(omega.shape) == 3:
        omega = pattern[0]
    orig_file = dir+'imgs_n1' +  '.npy'
    recovered = np.load(recdir + 'rec_noise'+str(int(noise_percent*100))+ '_p' + str(int(p*100)) + type + '_n' + str(num_samples) + '.npy')
    return fourier_file, orig_file, omega, recovered

def get_complex(dir, recdir, noise_percent, p, type):
    fourier_file, orig_file, omega, recovered = get_files(dir, recdir, noise_percent, p, type)
    linrec = linear_reconstruction(fourier_file, omega)
    venc = np.load(dir + 'venc_n1' + '.npy')
    orig = np.load(orig_file)
    new_shape = crop(orig[0,0,0]).shape
    kspace = getKspace(orig, venc)
    temp_save = './kspace_temp.npy'
    print('kspace',kspace.shape)
    np.save(temp_save, kspace)
    orig = linear_reconstruction(temp_save, np.zeros(new_shape, dtype=bool))
    o = np.zeros(recovered.shape, dtype=complex)
    for k in range(0,num_samples):
        o[k] = orig[0,:,:, :new_shape[0], :new_shape[1]]
    return recovered, linrec, o

def get_final(dir,recdir,noise_percent, p, type):
    fourier_file, orig_file, omega, recovered = get_files(dir, recdir, noise_percent, p, type)
    orig = np.load(orig_file) 
    new_shape = crop(orig[0,0,0]).shape
    linrec = linear_reconstruction(fourier_file, omega)
    venc = np.load(dir + 'venc_n1' + '.npy')
    imgs = recover_vel(linrec, venc)
    csimgs = recover_vel(recovered, venc)
    o = np.zeros(csimgs.shape, dtype=complex)
    for k in range(0,num_samples):
        o[k] = orig[0,:,:, :new_shape[0], :new_shape[1]]
    return csimgs, imgs, o

def get_error(dir, recdir, noise_percent, p, type,  use_complex):
    if use_complex:
        csimgs, linimgs, o = get_complex(dir,recdir, noise_percent, p, type)
    else:
        csimgs, linimgs, o = get_final(dir,recdir, noise_percent, p, type) 
    print('orig', o.shape)
    print('cs', csimgs.shape) 
   # msecs = ((o-csimgs)**2).mean(axis=0)
   # mselin = ((o-imgs)**2).mean(axis=0)
    msecs = np.zeros(csimgs.shape[0])
    mselin = np.zeros(csimgs.shape[0])
    for k in range(csimgs.shape[0]):
        msecs[k] = np.abs(((o[k]-csimgs[k])**2).mean(axis=None))
        mselin[k] = np.abs(((o[k]-linimgs[k])**2).mean(axis=None))
    print('mse',msecs.shape, mselin.shape)
    return msecs, mselin
def get_folder(use_complex):    
    if use_complex:
        folder = 'histcomplex'
    else:
        folder = 'histfinal'
    return folder

def formatting(ax, lgd):
    plt.xlabel('MSE',fontsize=fs)
    plt.ylabel('Percent of total samples',fontsize=fs)
    plt.tick_params(labelsize=fs)
    plt.tight_layout()
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    x_formatter = ScalarFormatter(useOffset=False)
    ax.xaxis.set_major_formatter(x_formatter)
    plt.legend(['1\% noise', '5\% noise', '10\% noise', '30\% noise'])

def plotpdiff(dir, recdir, noise_percent, p, type, use_complex, useCS):
    folder = get_folder(use_complex)
    fig, ax = plt.subplots(figsize=(4,3))
    i = 0
    colors = ['blue','orange','green', 'red']
    alpha = 1
    for noise_percent in [0.01, 0.05, 0.1, 0.3]:#
        msecs, mselin = get_error(dir, recdir, noise_percent, p, type, use_complex)
        if useCS:
            toplot = msecs
            msg = 'hist_debias'
        else:
            toplot = mselin
            msg = 'hist_debias_lin'
        plt.hist(toplot, bins=10, density=False,weights=np.ones(len(toplot)) / len(toplot),edgecolor=colors[i],alpha=alpha)#, ec='black')
        i = i + 1
        alpha = alpha - 0.25
    formatting(['1\% noise', '5\% noise', '10\% noise', '30\% noise'])
    plt.savefig(recdir + folder + '/' + msg + '_p' + str(int(p*100)) + type + '.png')
    print(recdir + folder + '/' + msg + '_p' + str(int(p*100)) + type + '.png')
    #plt.savefig(recdir + 'histnonzero/hist_debias'+ '_noise' + str(int(noise_percent*100)) + type + '.png')
    #plt.draw()
    plt.close('all')

def plotnoisediff(dir, recdir, noise_percent, p, type, use_complex, useCS):
    folder = get_folder(use_complex)
    colors = ['blue','orange','green', 'red']
    fig, ax = plt.subplots(figsize=(4,3))
    i = 0
    alpha = 1
    for p in [0.25, 0.5, 0.75]: 
        msecs, mselin = get_error(dir, recdir, noise_percent, p, type, use_complex)
        if useCS:
            toplot = msecs
            msg = 'hist_debias'
        else:
            toplot = mselin
            msg = 'hist_debias_lin'
        plt.hist(toplot, bins=10, density=False,weights=np.ones(len(toplot)) / len(toplot),edgecolor=colors[i], alpha=alpha)# ec='black')
        i = i + 1
    formatting(ax, ['25\% undersampling', '50\% undersampling', '75\% undersampling'])
    plt.savefig(recdir + folder + '/' + msg + '_noise' + str(int(noise_percent*100)) + type + '.png')
    print(recdir + folder + '/' + msg + '_noise' + str(int(noise_percent*100)) + type + '.png')
    #plt.savefig(recdir + 'hist/hist_lin'+ '_noise' + str(int(noise_percent*100)) + '_p' + str(int(p*100)) + type + '.png')
    #plt.show()
    plt.close('all')

def plthist(dir,recdir, noise_percent, p, type, use_complex):
   
    msecs, mselin = get_error(dir, recdir, noise_percent, p, type, use_complex)
    plotnoisediff(dir, recdir, noise_percent, p, type, use_complex, True)
    plotnoisediff(dir, recdir, noise_percent, p, type, use_complex, False)
    #plotpdiff(dir, recdir, noise_percent, p, type, use_complex, True)
    #plotpdiff(dir, recdir, noise_percent, p, type, use_complex, False)
    

if __name__ == '__main__':
    dir =home + '/apps/undersampled/modelflow/idealmodel/npy/'
    #dir = home + "/apps/undersampled/poiseuille/npy/"
    #recdir = home + "/apps/undersampled/poiseuille/debiasing/"
    recdir = dir
    #plthist(dir, recdir, 0.01, 0.25, 'bernoulli',use_complex=False) 
    for p in [0.25]:#, 0.5, 0.75]:
        for type in ['vardengauss']:#'bernoulli', 'bpoisson']:
            for noise in [0.01, 0.05, 0.1, 0.3]:
                try:
                    plthist(dir, recdir, noise, p, type, use_complex=False) 
                except Exception as e:
                    print(e)
                    print('missing', noise, 'noise', p, 'p', type, 'type')
                    continue
