import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import math
from scipy.optimize import curve_fit
from correlation import get_vals
sys.path.append('../../')
from recover import solver_folder
home = os.getenv('HOME')
fs=12
plt.rc('font',  family='serif')
plt.rc('xtick', labelsize='x-small')
plt.rc('ytick', labelsize='x-small')
plt.rc('text',  usetex=True)
start = 0
end = 15 #last distance to include in plot (final x axis value)
interval = int((end-start)/4)#int(math.ceil(((end-start)/4) / 10.0)) * 10))

def formatting(lgd,max):
    plt.xlabel('Distance',fontsize=fs)
    plt.ylabel('Correlation Coefficient',fontsize=fs)
    plt.tick_params(labelsize=fs)
    plt.ylim(top=max)
    plt.xticks(np.arange(start+1, end+1,interval))
    plt.tight_layout()

def get_coeff(noise_percent, p, samptype, n, size, num_pts, dir, ptsdir=None, kspacedir=None):
    if ptsdir is None:
        ptsdir = dir
    if kspacedir is None:
        kspacedir = dir
    corrfile = dir + 'plots/corrsqravg' + str(num_pts) + '_noise' + str(int(noise_percent*100)) + '_p' + str(int(p*100)) + samptype +'_n'+str(n) + '.npy'
    if os.path.exists(corrfile):
        #print('Correlation coefficient file exists.')
        coeff = np.load(corrfile)
    else:
        print('Calculating correlation coefficients...')
        coeff = get_vals(noise_percent, p, samptype, n, size, num_pts, recdir=dir, kspacedir=kspacedir, ptsdir=ptsdir, save_numpy=False)
    return coeff

def plot_corr_vplt(noise_percent, p, samptype, n, size, num_pts, v, dir, ptsdir, kspacedir):
    coeff = get_coeff(noise_percent, p, samptype, n, size, num_pts, dir, ptsdir, kspacedir)
    coeff = coeff[v]
    coeff[np.isnan(coeff)] = 0
    print('max coeff', np.amax(coeff))
    print(coeff[start:end].shape)
    p = plt.violinplot(np.swapaxes(coeff[start:end], 0,1))
    return p['cbars']

def plot_corr(noise_percent, p, samptype, n, size, num_pts, v, dir, ptsdir, kspacedir):
    coeff = get_coeff(noise_percent, p, samptype, n, size, num_pts, dir, ptsdir, kspacedir)
    coeff = coeff[v]
    coeff[np.isnan(coeff)] = 0
    corravg = np.mean(coeff, axis=1)
    corrmin = np.percentile(coeff, 10, axis=1)
    corrmax = np.percentile(coeff, 90, axis=1)
    size = len(corravg)
    p, = plt.plot(range(start+1,end+1), corravg[start:end])
    plt.fill_between(range(start+1,end+1), corrmin[start:end], corrmax[start:end], alpha=0.2)
    return p

def get_max(coeff, mx, v):
    coeff = coeff[v]            
    corravg = np.mean(coeff, axis=1)
    corrmax = np.percentile(coeff, 90, axis=1)
    return max(mx, np.amax(corrmax))

def find_max(noise_vals, p_vals, samp_vals, noise_val, p_val, samp_val, n, size, num_pts, dir, ptsdir, kspacedir):
    print("Finding a maximum correlation for all plots...")
    mx=0.
    coeff = get_coeff(noise_val, p_val, samp_val, n, size, num_pts, dir, ptsdir, kspacedir)
    maxv = coeff.shape[0]
    for v in range(0,maxv):
        for noise_percent in noise_vals:
            coeff = get_coeff(noise_percent, p_val, samp_val, n, size, num_pts, dir, ptsdir, kspacedir)
            mx = get_max(coeff, mx, v)
    for v in range(0,maxv):
        for p in p_vals:
            coeff = get_coeff(noise_val, p, samp_val, n, size, num_pts, dir, ptsdir, kspacedir)
            mx = get_max(coeff, mx, v)
    for v in range(0,maxv):
        for samptype in samp_vals:
            coeff = get_coeff(noise_val, p_val, samptype, n, size, num_pts, dir, ptsdir, kspacedir)
            mx = get_max(coeff, mx, v)
    return mx

def plot_noisediff(noise_vals, p, samptype, n, size, num_pts, max, dir, ptsdir, kspacedir, save_fig=True):
    print("Plotting noise percent comparison with baseline", p, samptype)
    coeff = get_coeff(noise_vals[0], p, samptype, n, size, num_pts, dir, ptsdir, kspacedir)
    maxv = coeff.shape[0]
    for v in range(0,maxv):
        plt.figure(figsize=(4,3))
        count = 0
        handles = [None]*len(noise_vals)
        for noise_percent in noise_vals:
            handles[count] = plot_corr(noise_percent, p, samptype, n, size, num_pts, v, dir, ptsdir, kspacedir)
            count = count + 1
        lgd = [str(int(x*100)) + '\% noise' for x in noise_vals] 
        plt.legend(handles, lgd)
        formatting(lgd, max)
        if save_fig:
            folder = '/plots/correlation/'
            if not os.path.exists(dir + folder):
                os.makedirs(dir+folder)  
            plt.savefig(dir + folder+ '/diffnoise' + str(start) + 'to' + str(end) + '_p' + str(int(p*100)) + samptype + '_v' + str(v) + '.pdf')
            plt.close()
        else:
            plt.show()

def plot_pdiff(noise_percent, p_vals, samptype, n, size, num_pts, max, dir, ptsdir, kspacedir, save_fig=True):
    print("Plotting undersampling percent comparison with baseline", noise_percent, samptype)
    coeff = get_coeff(noise_percent, p_vals[0], samptype, n, size, num_pts, dir, ptsdir, kspacedir)
    maxv = coeff.shape[0]
    for v in range(0,maxv):
        plt.figure(figsize=(4,3))
        count = 0
        handles = [None]*len(noise_vals)
        for p in p_vals:
            handles[count] = plot_corr(noise_percent, p, samptype, n, size, num_pts, v, dir, ptsdir, kspacedir) 
            count = count + 1
        lgd = [str(int(x*100)) + '\% undersampling' for x in p_vals]#['25\% undersampling', '50\% undersampling', '75\% undersampling']
        plt.legend(handles, lgd)
        formatting(lgd, max)
        if save_fig:
            folder = '/plots/correlation/'
            if not os.path.exists(dir + folder):
                os.makedirs(dir+folder)
            plt.savefig(dir + folder +'/diffundersamp' + str(start) + 'to' + str(end) + '_noise' + str(int(noise_percent*100)) + '_' + samptype + '_v' + str(v) + '.pdf')
            plt.close()
        else:
            plt.draw()

def plot_sampdiff(noise_percent, p, samp_vals, n, size, num_pts, max, dir, ptsdir, kspacedir, save_fig=True):
    print("Plotting undersampling mask comparison with baseline", noise_percent, p)
    coeff = get_coeff(noise_percent, p, samp_vals[0], n, size, num_pts, dir, ptsdir, kspacedir)
    maxv = coeff.shape[0]
    for v in range(0,maxv):
        plt.figure(figsize=(4,3))
        count = 0
        handles = [None]*len(noise_vals)
        for samptype in samp_vals: #'bernoulli', 'halton', 'vardengauss','vardentri', 'vardenexp'
            handles[count] = plot_corr(noise_percent, p, samptype, n, size, num_pts, v, dir, ptsdir, kspacedir)
            count = count + 1
        lgd = [x.capitalize() + ' undersampling' for x in samp_vals]
        plt.legend(handles, lgd) 
        formatting(lgd, max)
        if save_fig:
            folder = '/plots/correlation/'
            if not os.path.exists(dir + folder):
                os.makedirs(dir+folder)
            plt.savefig(dir + folder + '/diffsamptype' + str(start) + 'to' + str(end) + '_noise' + str(int(noise_percent*100)) + '_p' + str(int(p*100)) + '_v' + str(v) + '.png')
            plt.close()
        else:
            plt.draw()

def plot_methoddiff(noise_percent, p, samptype, n, size, num_pts, max, recdir, ptsdir, kspacedir, save_fig=True):
    print("Plotting solver method comparison with baseline", noise_percent, p, samptype)
    coeff = get_coeff(noise_percent, p, samptype, n, size, num_pts, recdir+solver_folder(0), ptsdir, kspacedir)
    maxv = coeff.shape[0]
    for v in range(0,maxv):
        plt.figure(figsize=(4,3))
        count = 0
        #handles = [None]
        handles = [None]*3
        for methodfolder in [solver_folder(0), solver_folder(1), solver_folder(2)]: #'cs', 'csdebias', 'omp'
            print('folder', methodfolder)
            handles[count] = plot_corr(noise_percent, p, samptype, n, size, num_pts, v, recdir+methodfolder, ptsdir, kspacedir)
            count = count + 1
        lgd = ["CS", "CSDEBIAS", "OMP"]
        plt.legend(handles, lgd)
        formatting(lgd, max)
        if save_fig:
            folder = '/plots/correlation/'
            if not os.path.exists(recdir + folder):
                os.makedirs(recdir+folder)
            plt.savefig(recdir + folder + '/diffmethod' + str(start) + 'to' + str(end) + '_noise' + str(int(noise_percent*100)) + '_p' + str(int(p*100)) + samptype + '_v' + str(v) + '.pdf')
            plt.close()
        else:
            plt.draw()
            
if __name__ == '__main__':
    if len(sys.argv) > 1:
        #specify parameters that are common in comparison plots
        # e.g. plot each noise level (1%, 5%, 10%) comparison in one plot, where each has 75% Gaussian undersampling
        noise_percent = float(sys.argv[1])
        p = float(sys.argv[2]) 
        samptype = sys.argv[3] #bernoulli, vardengauss, 'bpoisson', 'halton','vardentri', 'vardenexp'
        numsamples = int(sys.argv[4])
        recdir = sys.argv[5]
    else:
        recdir = home + '/apps/undersampled/poiseuille/debiasing/'
        noise_percent = 0.1
        p = 0.75
        samptype = 'vardengauss'
        numsamples = 100
    if len(sys.argv) > 6:
        ptsdir = sys.argv[6]
        kspacedir = sys.argv[7]
        solver_mode = int(sys.argv[8])
    else:
        ptsdir = None
        kspacedir = None
        solver_mode = 0
    size = 200
    size = 50
    num_pts = 50
    noise_vals = [0.01, 0.05, 0.1, 0.3]
    p_vals = [0.25, 0.5, 0.75]
    samp_vals = ['bernoulli', 'vardengauss']#['bernoulli', 'halton', 'vardengauss', 'vardentri', 'vardenexp']
    print("Creating correlation plots...")
    mf = solver_folder(solver_mode)
    #max_corr = find_max(noise_vals, p_vals, samp_vals, noise_percent, p, samptype, numsamples, size, num_pts, recdir+mf, ptsdir, kspacedir)
    max_corr=1 #max for poiseuille axis2 and ideal flow
    print('max', max_corr)
    sf = True
    plot_noisediff(noise_vals, p, samptype, numsamples, size, num_pts, max_corr, recdir+mf, ptsdir, kspacedir, save_fig=sf)
    plot_pdiff(noise_percent, p_vals, samptype, numsamples, size, num_pts, max_corr, recdir+mf, ptsdir, kspacedir, save_fig=sf)
    plot_sampdiff(noise_percent, p, samp_vals, numsamples, size, num_pts, max_corr, recdir+mf, ptsdir, kspacedir, save_fig=sf)
    plot_methoddiff(noise_percent, p, samptype, numsamples, size, num_pts, max_corr, recdir, ptsdir, kspacedir, save_fig=sf)
    plt.show() 
