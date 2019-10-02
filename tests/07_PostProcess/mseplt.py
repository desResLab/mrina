import sys
import os
sys.path.append('../../')
from recover import linear_reconstruction, recover_vel, solver_folder #threshold for zero mag
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

def get_files(dir, recdir, noise_percent, p, type, num_samples):
    #patterndir = home + "/apps/undersampled/poiseuille/npy/"
    patterndir = dir
    fourier_file = dir + 'noisy_noise' + str(int(noise_percent*100)) + '_n' + str(num_samples) + '.npy'
    undersample_file = patterndir + 'undersamplpattern_p' + str(int(p*100)) + type +  '_n' + str(num_samples) + '.npy'
    pattern = np.load(undersample_file)
    omega = pattern
    if len(omega.shape) == 3:
        omega = pattern[0]
    orig_file = dir+'imgs_n1' +  '.npy'
    recovered = np.load(recdir + 'rec_noise'+str(int(noise_percent*100))+ '_p' + str(int(p*100)) + type + '_n' + str(num_samples) + '.npy')
    return fourier_file, orig_file, omega, recovered

def get_complex(dir, recdir, noise_percent, p, type, num_samples):
    fourier_file, orig_file, omega, recovered = get_files(dir, recdir, noise_percent, p, type, num_samples)
    linrec = linear_reconstruction(fourier_file, omega)
    venc = np.load(dir + 'venc_n1' + '.npy')
    orig = np.load(orig_file)
    new_shape = crop(orig[0,0,0]).shape
    kspace = getKspace(orig, venc)
    orig = linear_reconstruction(kspace, np.zeros(new_shape, dtype=bool))
    return recovered, linrec, orig[0,:,:,:new_shape[0], :new_shape[1]] 

def get_final(dir,recdir,noise_percent, p, type, num_samples):
    fourier_file, orig_file, omega, recovered = get_files(dir, recdir, noise_percent, p, type, num_samples)
    orig = np.load(orig_file) 
    new_shape = crop(orig[0,0,0]).shape
    linrec = linear_reconstruction(fourier_file, omega)
    venc = np.load(dir + 'venc_n1' + '.npy')
    imgs = recover_vel(linrec, venc)
    csimgs = recover_vel(recovered, venc)
    return csimgs, imgs, orig[0,:,:,:new_shape[0], :new_shape[1]]

def get_error(dir, recdir, noise_percent, p, type, num_samples, use_complex, use_truth):
    if use_complex:
        csimgs, linimgs, o = get_complex(dir,recdir, noise_percent, p, type, num_samples)
    else:
        csimgs, linimgs, o = get_final(dir,recdir, noise_percent, p, type, num_samples) 
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
    return msecs, mselin

def get_folder(use_complex):    
    if use_complex:
        folder = '/plots/msecomplex'
    else:
        folder = '/plots/msefinal'
    return folder

def formatting(ax, lgd, xdesc):
    plt.ylabel('MSE',fontsize=fs)
    plt.tick_params(labelsize=fs)
    ax.set_xticks(range(1, len(lgd)+1))
    ax.set_xticklabels(lgd)
    plt.xlabel(xdesc)
    plt.tight_layout()

def plotpdiff(dir, recdir, noise_percent, p, type, num_samples, use_complex, use_truth, useCS):
    folder = get_folder(use_complex)
    fig, ax = plt.subplots(figsize=(4,3))
    i = 0
    colors = ['blue','orange','green', 'red']
    alpha = 1
    allplt = [None]*4
    count = 0
    for noise_percent in [0.01, 0.05, 0.1, 0.3]:#
        msecs, mselin = get_error(dir, recdir, noise_percent, p, type, num_samples, use_complex, use_truth)
        if useCS:
            toplot = msecs
            msg = 'vplt'
        else:
            toplot = mselin
            msg = 'vplt_lin'
        if not use_truth:
            msg = msg + 'avg'
        alpha = alpha - 0.25
        allplt[i] = toplot
        i = i + 1
    bplts = plt.violinplot(allplt)
    for patch, color in zip(bplts['bodies'], colors):
        patch.set_facecolor(color)
    formatting(ax, ['1\%', '5\%', '10\%', '30\%'], 'Noise')
    if not os.path.exists(recdir + folder):
        os.makedirs(recdir+folder)
    plt.savefig(recdir + folder + '/' + msg + '_p' + str(int(p*100)) + type + '.png')
    print("Saved as " + recdir + folder + '/' + msg + '_p' + str(int(p*100)) + type + '.png')
    #plt.draw()
    plt.close(fig)

def plotnoisediff(dir, recdir, noise_percent, p, type, num_samples, use_complex, use_truth, useCS):
    folder = get_folder(use_complex)
    colors = ['blue','orange','green', 'red']
    fig, ax = plt.subplots(figsize=(4,3))
    i = 0
    alpha = 1
    allplt = [None]*3
    for p in [0.25, 0.5, 0.75]: 
        msecs, mselin = get_error(dir, recdir, noise_percent, p, type, num_samples, use_complex, use_truth)
        if useCS:
            toplot = msecs
            msg = 'vplt'
        else:
            toplot = mselin
            msg = 'vplt_lin'
        if not use_truth:
            msg = msg + 'avg'
        i = i + 1
        allplt[int(p/0.25)-1] = toplot
    bplts = plt.violinplot(allplt)#, ['25\%', '50\%' '75\%'])
    for patch, color in zip(bplts['bodies'], colors):
        patch.set_facecolor(color)
    formatting(ax, ['25\%', '50\%', '75\%'], 'Undersampling')
    if not os.path.exists(recdir + folder):
        os.makedirs(recdir+folder)
    #plt.savefig(recdir + folder + '/' + msg + '_noise' + str(int(noise_percent*100)) + type + '.png')
    print("Saved as " + recdir + folder + '/' + msg + '_noise' + str(int(noise_percent*100)) + type + '.png')
    plt.show()
    plt.close(fig)

def plotmethoddiff(dir, recdir, noise_percent, p, type, num_samples, use_complex, use_truth, useCS, method='cs'):
    folder = get_folder(use_complex)
    colors = ['blue','orange','green', 'red']
    fig, ax = plt.subplots(figsize=(4,3))
    i = 0
    alpha = 1
    allplt = [None]*3
    for methodfolder in [solver_folder(0), solver_folder(1), solver_folder(2)]: 
        msecs, mselin = get_error(dir, recdir+methodfolder, noise_percent, p, type, num_samples, use_complex, use_truth)
        if useCS:
            toplot = msecs
            msg = 'vplt'
        else:
            toplot = mselin
            msg = 'vplt_lin'
        if not use_truth:
            msg = msg + 'avg'
        allplt[i] = toplot
        i = i + 1
    bplts = plt.violinplot(allplt)
    for patch, color in zip(bplts['bodies'], colors):
        patch.set_facecolor(color)
    formatting(ax, ['CS', 'CSDEBIAS', 'OMP'], 'Solver')
    if not os.path.exists(recdir + folder):
        os.makedirs(recdir+folder)
    filename = recdir + folder + '/' + msg + '_noise' + str(int(noise_percent*100)) + '_p'+str(int(p*100)) + type + '.png'
    plt.savefig(filename)
    print("Saved as " + filename)
    #plt.show()
    plt.close(fig)

def pltviolin(dir, recdir, num_samples, use_complex, use_truth):
    #use_complex: compare against complex images or final recovered velocity images
    #use_truth: compare against true values or the average recovered images 
    for p in [0.25]:
        for type in ['vardengauss']:#'bernoulli', 'bpoisson']:
            for noise in [0.01, 0.05, 0.1, 0.3]:
                try: #plot comparison of undersampling % for CS and linear rec. images
                    plotnoisediff(dir, recdir, noise, p, type, num_samples, use_complex, use_truth, True)
                    plotnoisediff(dir, recdir, noise, p, type, num_samples, use_complex, use_truth, False)
                except Exception as e:
                    print(e)
                    print('Not found: recovered images with ', noise, 'noise', p, 'p', type, 'type')
                    continue
    for p in [0.25, 0.5, 0.75]:
        for type in ['vardengauss']:#'bernoulli', 'bpoisson']:
            for noise in [0.01]:
                try:
                    plotpdiff(dir, recdir, noise, p, type, num_samples, use_complex, use_truth, True)
                    plotpdiff(dir, recdir, noise, p, type, num_samples, use_complex, use_truth, False)
                except Exception as e:
                    print(e)
                    print("Not found: recovered images with ", noise, 'noise', p, 'p', type, 'type')
                    continue
    

if __name__ == '__main__':
    if len(sys.argv) > 1:
        numsamples = int(sys.argv[1])
        dir = sys.argv[2]
        recdir = sys.argv[3]
        solver_mode = int(sys.argv[4])
    else:
        numsamples=100
        dir = home + "/apps/undersampled/poiseuille/npy/"
        recdir = dir
        solver_mode = 0
    recdir = recdir + solver_folder(solver_mode)
    #to plot a single violin plot, use plotnoisediff or plotpdiff
    #plotnoisediff(dir, recdir, 0.1, 0.5, 'bernoulli', use_complex=True, use_truth=True, useCS=True)
    #to plot all combinations:
    print("Creating MSE violin plots...")
    pltviolin(dir, recdir, numsamples, use_complex=False, use_truth=False)
    pltviolin(dir, recdir, numsamples, use_complex=True, use_truth=False)
    pltviolin(dir, recdir, numsamples, use_complex=False, use_truth=True)
    pltviolin(dir, recdir, numsamples, use_complex=True, use_truth=True)
