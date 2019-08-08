import sys
from CSRecoverySuite import CSRecovery,CSRecoveryDebiasing,OMPRecovery, Operator4dFlow, pywt2array, array2pywt, crop
import cv2
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
    print('stdev', stdev)
    print('noise_percent', noise_percent)
    rv = norm()
    #ppf: inverse of cdf
    eta = stdev*math.sqrt(2*m + 2*math.sqrt(m)*rv.ppf(0.95))
    print('eta: ', eta)
    return eta

def recover(noisy, original, pattern, wsz, processnum, return_dict, wvlt, solver_mode=CS_MODE):
    def recover(kspace,imsz,eta,omega):
        print('shape',kspace.shape)
        #im = crop(im)      #crop for multilevel wavelet decomp. array transform
        wim          = pywt2array( pywt.wavedec2(fft.ifft2(kspace), wavelet=wvlt, mode='periodization'), imsz)
        wsz = wim.shape
        A = Operator4dFlow( imsz=imsz, insz=wsz, samplingSet=~omega, waveletName=wvlt, waveletMode='periodization' )
        yim          = A.eval( wim, 1 )
        print('l2 norm of yim: ', np.linalg.norm(yim.ravel(), 2))
        if solver_mode == OMP_MODE:
          wim = OMPRecovery(A, yim, showProgress=False)[0]
        else:
          wim =  CSRecovery(eta, yim, A, np.zeros( wsz ), disp=1, method='pgdl1')
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
    print('recov shape', cs.shape)
    if len(pattern.shape) > 2:
        omega = crop(pattern[0])
    else:
        omega = crop(pattern)
    print('omega', omega.shape)
    print('original',original.shape)
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
    print(shp)
    print(pattern.shape)
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
    print('num of processes:', len(jobs))
    for job in jobs:
        job.join()
    recovered = np.concatenate([v for k,v in sorted(return_dict.items())], axis=0)
    print('recovered shape', recovered.shape)
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

def linear_reconstruction(fourier_file, omega):
    kspace = np.load(fourier_file)
    omega = crop(omega)
    kspace = kspace[:,:,:, :omega.shape[0], :omega.shape[1]]
    linrec = np.zeros(kspace.shape[0:3] + omega.shape, dtype=complex)
    for n in range(kspace.shape[0]):
        for k in range(kspace.shape[1]):
            for j in range(kspace.shape[2]):
                kspace[n,k,j][omega] = 0
                linrec[n,k,j] = fft.ifft2(crop(kspace[n,k,j]))
    return linrec

if __name__ == '__main__':
    if len(sys.argv) > 1:
        noise_percent = float(sys.argv[1])
        p=float(sys.argv[2])
        type=sys.argv[3]
        num_samples = int(sys.argv[4])
    else:
        noise_percent=0.01
        p=0.75 #percent not sampled
        type='bernoulli'
        num_samples = 100
    save_img = False #whether to save example image files
    dir = home + '/apps/undersampled/poiseuille/npy/'#where the kspace data is
    recdir = dir #where to save recovered imgs
    patterndir = home + '/apps/undersampled/poiseuille/npy/' #where the undersampling patterns are located
    num_processes = 2
    wavelet_type = 'haar'
    solver_mode = CS_MODE    
 
    fourier_file = dir + 'noisy_noise' + str(int(noise_percent*100)) + '_n' + str(num_samples) + '.npy'
    undersample_file = patterndir + 'undersamplpattern_p' + str(int(p*100)) + type +  '_n' + str(num_samples) + '.npy'
    pattern = np.load(undersample_file)
    omega = pattern[0]
    orig_file = dir+'imgs_n1' +  '.npy'
    recovered = recoverAll(fourier_file, orig_file, pattern, c=num_processes, wvlt=wavelet_type, mode=solver_mode)
    np.save(recdir + 'rec_noise'+str(int(noise_percent*100))+ '_p' + str(int(p*100)) + type  + '_n' + str(num_samples), recovered)
    #recovered = np.load(recdir + 'rec_noise'+str(int(noise_percent*100))+ '_p' + str(int(p*100)) + type + '_n' + str(num_samples) + '.npy')
    print('recovered images', recovered.shape)
    linrec = linear_reconstruction(fourier_file, omega)
    venc = np.load(dir + 'venc_n1' + '.npy')
    imgs = recover_vel(linrec, venc)
    csimgs = recover_vel(recovered, venc)
    orig = np.load(orig_file)
    new_shape = crop(orig[0,0,0]).shape
    
    if save_img:
        scipy.misc.imsave(recdir + 'csmag.jpg', np.abs(csimgs[0,0,0]))
        scipy.misc.imsave(recdir+'linmag.jpg', np.abs(imgs[0,0,0]))
        scipy.misc.imsave(recdir+'origmag.jpg', np.abs(orig[0,0,0]))
        for k in range(1,4):
            scipy.misc.imsave(recdir+'csvel'+str(k) + '.jpg', np.abs(csimgs[0,k,0]))
            scipy.misc.imsave(recdir+'linvel'+str(k) + '.jpg', np.abs(imgs[0,k,0]))
            scipy.misc.imsave(recdir+'origvel'+str(k) + '.jpg', np.abs(orig[0,k,0]))
    
    o = np.zeros(csimgs.shape, dtype=complex)
    for k in range(0,num_samples):
        o[k] = orig[0,:, :new_shape[0], :new_shape[1]]
    print(o.shape)
    print(csimgs.shape)
    print(linrec.shape)
    print('mse between original and recovered images: ', (np.square(csimgs - o[:,:,:,:new_shape[0], :new_shape[1]])).mean())
    print('mse between original and linear reconstructed images: ', (np.square(imgs - o[:,:,:,:new_shape[0], :new_shape[1]])).mean())
