from CSRecoverySuite import CSRecovery, Operator4dFlow, pywt2array, array2pywt, crop
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

home = os.getenv('HOME')

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

def recover(noisy, original, wsz, processnum, return_dict):
    def recover(kspace,imsz,eta,omega):
        print('shape',kspace.shape)
        #im = crop(im)      #crop for multilevel wavelet decomp. array transform
        wim          = pywt2array( pywt.wavedec2(fft.ifft2(kspace), wavelet='haar', mode='periodic'), imsz)
        wsz = wim.shape
        A = Operator4dFlow( imsz=imsz, insz=wsz, samplingSet=~omega, waveletName='haar', waveletMode='periodic' )
        yim          = A.eval( wim, 1 )
        print('l2 norm of yim: ', np.linalg.norm(yim.ravel(), 2))
        cswim =  CSRecovery(eta, yim, A, np.zeros( wsz ), disp=1, method='pgdl1',maxItns=4)
        if isinstance(cswim, tuple):
            cswim = cswim[0] #for the case where ynrm is less than eta
        csim    = pywt.waverec2(array2pywt( cswim ), wavelet='haar', mode='periodic')
        return csim
    cs = np.zeros(noisy.shape,dtype=complex)
    imsz = noisy.shape[3:]
    for n in range(0,noisy.shape[0]):
        omega = crop(pattern[n]);
        print('omega', omega.shape)
        for k in range(0,4):
            for j in range(0, noisy.shape[2]):
                im = original[n,k,j]
                imNrm=np.linalg.norm(im.ravel(), 2)
                eta = get_eta(im, imNrm, noise_percent, imsz[0])
                cs[n,k,j] = recover(noisy[n,k,j], imsz, eta, omega)

    return_dict[processnum] = cs
    return cs

def recoverAll(fourier_file, orig_file, pattern, c=1):
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
    imsz = original[0,0,0].shape
    wsz = pywt2array(pywt.wavedec2(original[0,0,0], wavelet='haar', mode='periodic'), imsz).shape
    for n in range(0, data.shape[0], interval):
        p = Process(target=recover, args=(data[n:n+interval], original[n:n+interval], wsz, int(n/interval), return_dict))
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
    mag = np.abs(mag)
    return np.concatenate((np.expand_dims(mag, axis=1),vel), axis=1)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        noise_percent = float(sys.argv[1])
        p=float(sys.argv[2])
        type=sys.argv[3]
        num_samples = int(sys.argv[4])
    else:
        noise_percent=0.05
        p=0.10 #percent not sampled
        type='bernoulli'
        num_samples = 100
    dir = home + '/Documents/undersampled/poiseuille/npy/'
    fourier_file = dir + 'noisy_noise' + str(int(noise_percent*100)) + '_n' + str(num_samples) + '.npy'
    undersample_file = dir + 'undersamplpattern_p' + str(int(p*100)) + type +  '_n' + str(num_samples) + '.npy'
    pattern = np.load(undersample_file)
    orig_file = dir+'imgs_n' + str(num_samples) + '.npy'
    recovered = recoverAll(fourier_file, orig_file, pattern, c=2)
    np.save(dir + 'rec_noise'+str(int(noise_percent*100))+ '_p' + str(int(p*100)) + type  + '_n' + str(num_samples), recovered)
    #recovered = np.load(dir + 'rec_noise'+str(int(noise_percent*100))+ '_p' + str(int(p*100)) + type + '_n' + str(num_samples) + '.npy')
    print(recovered.shape)
    venc = np.load(dir + 'venc_n' + str(num_samples) + '.npy')
    imgs = recover_vel(recovered, venc)
    orig = np.load(orig_file)
    print('mse between original and recovered images: ', (np.square(imgs - orig)).mean())
