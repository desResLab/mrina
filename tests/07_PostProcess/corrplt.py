import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import math
from scipy.optimize import curve_fit
from correlation import get_vals
home = os.getenv('HOME')
fs=12
plt.rc('font',  family='serif')
plt.rc('xtick', labelsize='x-small')
plt.rc('ytick', labelsize='x-small')
plt.rc('text',  usetex=True)
start = 0
end = 30 #last distance to include in plot (final x axis value)
interval = int((end-start)/4)#int(math.ceil(((end-start)/4) / 10.0)) * 10))

def formatting(lgd,max):
    plt.legend(lgd)
    plt.xlabel('Distance',fontsize=fs)
    plt.ylabel('Correlation Coefficient ($R^2$)',fontsize=fs)
    plt.tick_params(labelsize=fs)
    #plt.xticks(np.arange(start, end+1, 50))
    plt.ylim(top=max)
    plt.xticks(np.arange(start+1, end+1,interval))
    plt.tight_layout()

def get_coeff(noise_percent, p, samptype, n, size, num_pts, dir, ptsdir=None, kspacedir=None):
    if ptsdir is None:
        ptsdir = dir
    if kspacedir is None:
        kspacedir = dir
    corrfile = dir + 'results/corrsqravg' + str(num_pts) + '_noise' + str(int(noise_percent*100)) + '_p' + str(int(p*100)) + samptype +'_n'+str(n) + '.npy'
    if os.path.exists(corrfile):
        coeff = np.load(corrfile)
    else:
        print('retrieving correlation coeff', noise_percent, p, samptype)
        coeff = get_vals(noise_percent, p, samptype, n, size, num_pts, recdir=dir, kspacedir=kspacedir, ptsdir=ptsdir, save_numpy=False)
    return coeff

def plot_corr(noise_percent, p, samptype, n, size, num_pts, v, dir, ptsdir, kspacedir):
    coeff = get_coeff(noise_percent, p, samptype, n, size, num_pts, dir, ptsdir, kspacedir)
    coeff = coeff[v]
    corravg = np.mean(coeff, axis=1)
    corrmin = np.percentile(coeff, 10, axis=1)
    corrmax = np.percentile(coeff, 90, axis=1)
    size = len(corravg)
    plt.plot(range(start+1,end+1), np.abs(corravg)[start:end])
    plt.fill_between(range(start+1,end+1), corrmin[start:end], corrmax[start:end], alpha=0.2)

def get_max(coeff, mx, v):
    coeff = coeff[v]            
    corravg = np.mean(coeff, axis=1)
    corrmax = np.percentile(coeff, 90, axis=1)
    return max(mx, np.amax(corrmax))

def find_max(noise_vals, p_vals, samp_vals, noise_val, p_val, samp_val, n, size, num_pts, dir, ptsdir, kspacedir):
    mx=0.
    for v in range(0,4):
        plt.figure(figsize=(4,3))
        for noise_percent in noise_vals:
            coeff = get_coeff(noise_percent, p_val, samp_val, n, size, num_pts, dir, ptsdir, kspacedir)
            mx = get_max(coeff, mx, v)
    for v in range(0,4):
        plt.figure(figsize=(4,3))
        for p in p_vals:
            coeff = get_coeff(noise_val, p, samp_val, n, size, num_pts, dir, ptsdir, kspacedir)
            mx = get_max(coeff, mx, v)
    for v in range(0,4):
        plt.figure(figsize=(4,3))
        for samptype in samp_vals:
            coeff = get_coeff(noise_val, p_val, samptype, n, size, num_pts, dir, ptsdir, kspacedir)
            mx = get_max(coeff, mx, v)
    return mx

def plot_noisediff(noise_vals, p, samptype, n, size, num_pts, max, dir, ptsdir, kspacedir, save_fig=True):
    for v in range(0,4):
        plt.figure(figsize=(4,3))
        for noise_percent in noise_vals:#[0.01,0.05,0.1,0.3]:
            plot_corr(noise_percent, p, samptype, n, size, num_pts, v, dir, ptsdir, kspacedir)
        lgd = [str(x) + '\% noise' for x in noise] 
        formatting(lgd, max)
        if save_fig:
            plt.savefig(dir + 'results/diffnoise' + str(start) + 'to' + str(end) + '_p' + str(int(p*100)) + '_v' + str(v) + '.png')
        else:
            plt.draw()

def plot_pdiff(noise_percent, p_vals, samptype, n, size, num_pts, max, dir, ptsdir, kspacedir, save_fig=True):
    for v in range(0,4):
        plt.figure(figsize=(4,3))
        for p in p_vals:#[0.25,0.5, 0.75]:
            plot_corr(noise_percent, p, samptype, n, size, num_pts, v) 
        lgd = [str(x) + '\% undersampling' for x in p_vals]#['25\% undersampling', '50\% undersampling', '75\% undersampling']
        formatting(lgd, max)
        if save_fig:
            plt.savefig(dir + 'results/diffundersamp' + str(start) + 'to' + str(end) + '_noise' + str(int(noise_percent*100)) + '_v' + str(v) + '.png')
        else:
            plt.draw()

def plot_sampdiff(noise_percent, p, samp_vals, n, size, num_pts, max, v, dir, ptsdir, kspacedir):
    for v in range(0,4):
        plt.figure(figsize=(4,3))
        for samptype in ['bernoulli']:#, 'bpoisson', 'halton', 'vardengauss','vardentri', 'vardenexp']:
            plot_corr(noise_percent, p, samptype, n, size, num_pts, v)
        lgd = ['Bernoulli undersampling','Poisson undersampling', 'Halton undersampling', 'Gauss density undersampling', 'Tri density undersampling','Exp density undersampling']
        formatting(lgd, max)
        if save_fig:
            plt.savefig(dir + 'results/diffsamptype' + str(start) + 'to' + str(end) + '_noise' + str(int(noise_percent*100)) + '_p' + str(int(p*100)) + '_v' + str(v) + '.png')
        else:
            plt.draw()

if __name__ == '__main__':
    if len(sys.argv) > 1:
        #specify parameters that are common in comparison plots
        # e.g. plot each noise level (1%, 5%, 10%) comparison in one plot, where each has 75% Gaussian undersampling
        noise_percent = float(sys.argv[1])
        p = float(sys.argv[2]) 
        samptype = sys.argv[3] #bernoulli, vardengauss, 'bpoisson', 'halton','vardentri', 'vardenexp'
        dir = sys.argv[4]
    else:
        dir = home + '/apps/undersampled/poiseuille/debiasing/'
        noise_percent = 0.1
        p = 0.75
        samptype = 'vardengauss'
    if len(sys.argv) > 5:
        ptsdir = sys.argv[5]
        kspacedir = sys.argv[6]
    else:
        ptsdir = None
        kspacedir = None
    noise_vals = [0.01, 0.05, 0.1, 0.3]
    p_vals = [0.25, 0.5, 0.75]
    samp_vals = ['bernoulli', 'vardengauss']
    max_corr = find_max(noise_vals, p_vals, samp_vals, noise_percent, p, samptype, n, size, num_pts, dir, ptsdir, kspacedir)
    plot_noisediff(noise_vals, p, samptype, size, num_pts, max_corr, dir, ptsdir, kspacedir)
    plot_pdiff(noise_percent, p_vals, samptype, size, num_pts, max_corr, dir, ptsdir, kspacedir)
    plot_sampdiff(noise_percent, p, samp_vals, size, num_pts, max_corr, dir, ptsdir, kspacedir)

    
