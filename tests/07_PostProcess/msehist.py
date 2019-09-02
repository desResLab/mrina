import sys
import os
sys.path.append('../../')
from recover import linear_reconstruction, recover_vel #threshold for zero mag
from genSamples import getKspace
import numpy as np
import matplotlib.pyplot as plt
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
    orig = linear_reconstruction(kspace, np.zeros(new_shape, dtype=bool))
    return recovered, linrec, orig[0,:,:,:new_shape[0], :new_shape[1]] 

def get_final(dir,recdir,noise_percent, p, type):
    fourier_file, orig_file, omega, recovered = get_files(dir, recdir, noise_percent, p, type)
    orig = np.load(orig_file) 
    new_shape = crop(orig[0,0,0]).shape
    linrec = linear_reconstruction(fourier_file, omega)
    venc = np.load(dir + 'venc_n1' + '.npy')
    imgs = recover_vel(linrec, venc)
    csimgs = recover_vel(recovered, venc)
    return csimgs, imgs, orig[0,:,:,:new_shape[0], :new_shape[1]]

def get_error(dir, recdir, noise_percent, p, type,  use_complex, use_truth):
    if use_complex:
        csimgs, linimgs, o = get_complex(dir,recdir, noise_percent, p, type)
    else:
        csimgs, linimgs, o = get_final(dir,recdir, noise_percent, p, type) 
    print('orig', o.shape)
    print('cs', csimgs.shape) 
    avgcs = csimgs.mean(axis=0)
    avglin = linimgs.mean(axis=0)
    msecs = np.zeros(csimgs.shape[0])
    mselin = np.zeros(csimgs.shape[0])
    for k in range(csimgs.shape[0]):
        if use_truth:
            msecs[k] = np.abs(((o-csimgs[k])**2).mean(axis=None))
            mselin[k] = np.abs(((o-linimgs[k])**2).mean(axis=None))
        else: #compare against average recovered
            msecs[k] = np.abs(((avgcs-csimgs[k])**2).mean(axis=None))
            mselin[k] = np.abs(((avglin-linimgs[k])**2).mean(axis=None))
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
    plt.legend(lgd)

def plotpdiff(dir, recdir, noise_percent, p, type, use_complex, use_truth, useCS):
    folder = get_folder(use_complex)
    fig, ax = plt.subplots(figsize=(4,3))
    i = 0
    colors = ['blue','orange','green', 'red']
    alpha = 1
    for noise_percent in [0.01, 0.05, 0.1, 0.3]:#
        msecs, mselin = get_error(dir, recdir, noise_percent, p, type, use_complex, use_truth)
        if useCS:
            toplot = msecs
            msg = 'hist'
        else:
            toplot = mselin
            msg = 'hist_lin'
        if not use_truth:
            msg = msg + 'avg'
        plt.hist(toplot, bins=10, density=False,weights=np.ones(len(toplot)) / len(toplot),edgecolor=colors[i],alpha=alpha)#, ec='black')
        i = i + 1
        alpha = alpha - 0.25
    formatting(ax, ['1\% noise', '5\% noise', '10\% noise', '30\% noise'])
    plt.savefig(recdir + folder + '/' + msg + '_p' + str(int(p*100)) + type + '.png')
    print(recdir + folder + '/' + msg + '_p' + str(int(p*100)) + type + '.png')
    #plt.savefig(recdir + 'histnonzero/hist_debias'+ '_noise' + str(int(noise_percent*100)) + type + '.png')
    #plt.draw()
    plt.close('all')

def plotnoisediff(dir, recdir, noise_percent, p, type, use_complex, use_truth, useCS):
    folder = get_folder(use_complex)
    colors = ['blue','orange','green', 'red']
    fig, ax = plt.subplots(figsize=(4,3))
    i = 0
    alpha = 1
    for p in [0.25, 0.5, 0.75]: 
        msecs, mselin = get_error(dir, recdir, noise_percent, p, type, use_complex, use_truth)
        if useCS:
            toplot = msecs
            msg = 'hist'
        else:
            toplot = mselin
            msg = 'hist_lin'
        if not use_truth:
            msg = msg + 'avg'
        plt.hist(toplot, bins=10, density=False,weights=np.ones(len(toplot)) / len(toplot),edgecolor=colors[i], alpha=alpha)# ec='black')
        i = i + 1
    formatting(ax, ['25\% undersampling', '50\% undersampling', '75\% undersampling'])
    plt.savefig(recdir + folder + '/' + msg + '_noise' + str(int(noise_percent*100)) + type + '.png')
    print(recdir + folder + '/' + msg + '_noise' + str(int(noise_percent*100)) + type + '.png')
    #plt.savefig(recdir + 'hist/hist_lin'+ '_noise' + str(int(noise_percent*100)) + '_p' + str(int(p*100)) + type + '.png')
    #plt.show()
    plt.close('all')

def plthist(dir, recdir, use_complex, use_truth):
    #use_complex: compare against complex images or final recovered velocity images
    #use_truth: compare against true values or the average recovered images 
    for p in [0.25]:
        for type in ['vardengauss']:#'bernoulli', 'bpoisson']:
            for noise in [0.01, 0.05, 0.1, 0.3]:
                try: #plot comparison of undersampling % for CS and linear rec. images
                    plotnoisediff(dir, recdir, noise, p, type, use_complex, use_truth, True)
                    plotnoisediff(dir, recdir, noise, p, type, use_complex, use_truth, False)
                except Exception as e:
                    print(e)
                    print('missing', noise, 'noise', p, 'p', type, 'type')
                    continue
    for p in [0.25, 0.5, 0.75]:
        for type in ['vardengauss']:#'bernoulli', 'bpoisson']:
            for noise in [0.01]:
                try:
                    plotpdiff(dir, recdir, noise, p, type, use_complex, use_truth, True)
                    plotpdiff(dir, recdir, noise, p, type, use_complex, use_truth, False)
                except Exception as e:
                    print(e)
                    print('missing', noise, 'noise', p, 'p', type, 'type')
                    continue
    

if __name__ == '__main__':
    dir =home + '/apps/undersampled/modelflow/idealmodel/npy/'
    #dir = home + "/apps/undersampled/poiseuille/npy/"
    #recdir = home + "/apps/undersampled/poiseuille/debiasing/"
    recdir = dir
    #to plot a single histogram, use plotnoisediff or plotpdiff
    #plotnoisediff(dir, recdir, 0.1, 0.5, 'bernoulli', use_complex=True, use_truth=True, useCS=True)
    #to plot all combinations: 
    plthist(dir, recdir, use_complex=False, use_truth=False)
