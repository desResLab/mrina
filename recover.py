import sys
from CSRecoverySuite import CSRecovery,CSRecoveryDebiasing,OMPRecovery, Operator4dFlow, pywt2array, array2pywt, crop
import matplotlib.pyplot as plt
import numpy.fft as fft
import pywt
import numpy as np
import os
import math
import sys
from multiprocessing import Process, cpu_count, Manager
from genSamples import getKspace,getVenc
from scipy.stats import norm,bernoulli
import scipy.misc
home = os.getenv('HOME')

CS_MODE = 0
DEBIAS_MODE = 1
OMP_MODE = 2

def get_eta(im, imNrm, noise_percent, m):
    avgnorm = imNrm/math.sqrt(im.size)
    stdev = noise_percent * avgnorm
    rv = norm()
    #ppf: inverse of cdf
    eta = stdev*math.sqrt(2*m + 2*math.sqrt(m)*rv.ppf(0.95))
    if eta < 1E-3: #min. threshold for eta
        eta = 1E-3
    return eta

def recover(noisy, original, pattern, wsz, processnum, return_dict, wvlt, solver_mode=CS_MODE):
    def recover(kspace,imsz,eta,omega):
        wim          = pywt2array( pywt.wavedec2(fft.ifft2(kspace), wavelet=wvlt, mode='periodization'), imsz)
        wsz = wim.shape
        A = Operator4dFlow( imsz=imsz, insz=wsz, samplingSet=~omega, waveletName=wvlt, waveletMode='periodization' )
        yim          = A.eval( wim, 1 )
        if solver_mode == OMP_MODE:
          tol = eta/np.linalg.norm(yim.ravel(),2)
          print('Recovering using OMP with tol =', tol)
          wim = OMPRecovery(A, yim, tol=tol, showProgress=True, progressInt=250, maxItns=2000)[0]
        else:
          print('Recovering using CS with eta =', eta)
          wim =  CSRecovery(eta, yim, A, np.zeros( wsz ), disp=1)
        if isinstance(wim, tuple):
            wim = wim[0] #for the case where ynrm is less than eta
        if solver_mode == DEBIAS_MODE:
            wim =  CSRecoveryDebiasing( yim, A, wim)
            if isinstance(wim, tuple):
                wim = wim[0] #for the case where ynrm is less than eta
        csim    = pywt.waverec2(array2pywt( wim ), wavelet=wvlt, mode='periodization')
        return csim
    imsz = crop(noisy[0,0,0]).shape
    cs = np.zeros(noisy.shape[0:3] + imsz,dtype=complex)
    if len(pattern.shape) > 2:
        omega = crop(pattern[0])
    else:
        omega = crop(pattern)
    for n in range(0,noisy.shape[0]):
        for k in range(0,4):
            for j in range(0, noisy.shape[2]):
                im = crop(original[0,k,j])
                imNrm=np.linalg.norm(im.ravel(), 2)
                eta = get_eta(im, imNrm, noise_percent, imsz[0])
                cs[n,k,j] = recover(crop(noisy[n,k,j]), imsz, eta, omega)

    return_dict[processnum] = cs
    return cs

def recoverAll(fourier_file, orig_file, pattern, c=1, wvlt='haar', mode=CS_MODE):
	# Load data
    data = np.load(fourier_file)
    original = np.load(orig_file)
    shp = data.shape[0:3] + crop(np.zeros(data.shape[3:])).shape
    recovered = np.empty(shp,dtype=np.complex64)
    interval = max(int(data.shape[0]/c),1)
    manager = Manager()
    return_dict = manager.dict()
    jobs = []
    imsz = crop(original[0,0,0]).shape
    first = original[0,0,0]
    wsz = pywt2array(pywt.wavedec2(crop(first), wavelet=wvlt, mode='periodization'), imsz).shape
    for n in range(0, data.shape[0], interval):
        p = Process(target=recover, args=(data[n:n+interval], original, pattern, wsz, int(n/interval), return_dict, wvlt, mode))
        jobs.append(p)
        p.start()
    for job in jobs:
        job.join()
    recovered = np.concatenate([v for k,v in sorted(return_dict.items())], axis=0)
    print('Finished recovering, with final shape', recovered.shape)
    return recovered

def recover_vel(recovered, venc):
    mag = recovered[:, 0, :]
    vel = np.empty((recovered.shape[0],) + (3,) + recovered.shape[2:] )
    for n in range(0,recovered.shape[0]):
        for k in range(1,4):
            for j in range(0, recovered.shape[2]):
                m = mag[n,j]
                v = recovered[n,k,j]
                v = venc/(2*math.pi)*np.log(np.divide(v,m)).imag
                vel[n,k-1,j] = v
                #set velocity = 0 wherever mag is close enough to 0
                mask = (np.abs(mag[n,j]) < 1E-1)
                vel[n,k-1,j, mask] = 0
    mag = np.abs(mag)
    return np.concatenate((np.expand_dims(mag, axis=1),vel), axis=1)

def linear_reconstruction(fourier_file, omega=None):
    if isinstance(fourier_file, str): 
        kspace = np.load(fourier_file)
    else:
        kspace = fourier_file
    if omega is not None:
        if len(omega.shape) > 2:
            omega = crop(omega[0])
        else:
            omega = crop(omega)
    imsz = crop(kspace[0,0,0]).shape
    kspace = kspace[:,:,:, :imsz[0], :imsz[1]]
    linrec = np.zeros(kspace.shape[0:3] + imsz, dtype=complex)
    for n in range(kspace.shape[0]):
        for k in range(kspace.shape[1]):
            for j in range(kspace.shape[2]):
                if omega is not None:
                    kspace[n,k,j][omega] = 0
                linrec[n,k,j] = fft.ifft2(crop(kspace[n,k,j]))
    return linrec

def solver_folder(solver_mode):
    if solver_mode == CS_MODE:
        folder = 'cs/'
    elif solver_mode == DEBIAS_MODE:
        folder = 'csdebias/'
    elif solver_mode == OMP_MODE:
        folder = 'omp/'
    else:  
        print('ERROR: Invalid solver mode')
        sys.exit(-1)
    return folder

if __name__ == '__main__':
    if len(sys.argv) > 1:
        noise_percent = float(sys.argv[1])
        p=float(sys.argv[2])
        type=sys.argv[3]
        num_samples = int(sys.argv[4])
        num_processes = int(sys.argv[5])
        kspacedir = sys.argv[6]
        recdir = kspacedir
        patterndir = kspacedir 
        solver_mode = CS_MODE
        if len(sys.argv) > 8:
            recdir = sys.argv[7] 
            patterndir = sys.argv[8]
            solver_mode = int(sys.argv[9])
    else:
        noise_percent=0.01
        p=0.75 #percent not sampled
        type='bernoulli'
        num_samples = 100
        kspacedir = home + '/apps/undersampled/poiseuille/npy/'#where the kspace data is
        recdir = kspacedir #where to save recovered imgs
        patterndir = home + '/apps/undersampled/poiseuille/npy/' #where the undersampling patterns are located
        num_processes = 2
        solver_mode = CS_MODE 
    wavelet_type = 'haar'
    
    fourier_file = kspacedir + 'noisy_noise' + str(int(noise_percent*100)) + '_n' + str(num_samples) + '.npy'
    undersample_file = patterndir + 'undersamplpattern_p' + str(int(p*100)) + type +  '_n' + str(num_samples) + '.npy'
    pattern = np.load(undersample_file)
    omega = pattern[0]
    orig_file = kspacedir+'imgs_n1' +  '.npy'
    venc = np.load(kspacedir + 'venc_n1' + '.npy')
    folder = solver_folder(solver_mode)
    savednpy = recdir +folder+'rec_noise'+str(int(noise_percent*100))+ '_p' + str(int(p*100)) + type + '_n' + str(num_samples) + '.npy' 
    if not os.path.exists(savednpy):
        recovered = recoverAll(fourier_file, orig_file, pattern, c=num_processes, wvlt=wavelet_type, mode=solver_mode)
        if not os.path.exists(recdir+folder):
            os.makedirs(recdir+folder)
        np.save(savednpy, recovered)
        print('Recovered images saved as ', savednpy)
    else:
        print('Retrieving recovered images from numpy file', savednpy)
        recovered = np.load(savednpy)
    linrec = linear_reconstruction(fourier_file, omega)
    imgs = recover_vel(linrec, venc)
    csimgs = recover_vel(recovered, venc)
    orig = np.load(orig_file)
    new_shape = crop(orig[0,0,0]).shape
    
    o = np.zeros(csimgs.shape, dtype=complex)
    for k in range(0,num_samples):
        o[k] = orig[0,:, :new_shape[0], :new_shape[1]]
    print('MSE between original and recovered images: ', (np.square(csimgs - o[:,:,:,:new_shape[0], :new_shape[1]])).mean())
    print('MSE between original and linear reconstructed images: ', (np.square(imgs - o[:,:,:,:new_shape[0], :new_shape[1]])).mean())
