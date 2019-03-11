from CSRecoverySuite import CSRecovery, Operator4dFlow, pywt2array, array2pywt, crop
import cv2
import matplotlib.pyplot as plt
import numpy.fft as fft
import pywt
import numpy as np
import os
import math
from multiprocessing import Process
home = os.getenv('HOME')

def recover(fourier_file, pattern):
	# Load data
    data = np.load(fourier_file)
    #pattern = np.load(undersample_file)
    shp = data.shape[0:3] + crop(np.zeros(data.shape[3:])).shape
    recovered = np.empty(shp,dtype=np.complex64)
    print(shp)
    print(pattern.shape)

    for n in range(0,data.shape[0]):
        omega = crop(pattern[n]);
        for k in range(0,4):
            for j in range(0, data.shape[2]):

                im      = data[n,k,j];
                im = crop(im)      #crop for multilevel wavelet decomp. array transform
                imsz    = im.shape;

                # Wavelet Parameters
                wim         = pywt2array( pywt.wavedec2(im, wavelet='haar', mode='periodic'), imsz);
                wsz         = wim.shape;

                # 4dFlow Operator
                A           = Operator4dFlow( imsz=imsz, insz=wsz, samplingSet=omega, waveletName='haar', waveletMode='periodic' );
                # True data (recall A takes as input wavelet coefficients)
                yim         = A.eval( wim, 1 );

                # Recovery via CSRecovery
                #   Here we choose \eta as a fraction of the true image. In general it is
                #   proportional to the noise variance
                imNrm           = np.linalg.norm(im.ravel(), 2);
                eta             = 1E-3 * imNrm;
                cswim, fcwim    = CSRecovery(eta, yim, A, np.zeros( wsz ), disp=0, method='pgdl1', maxItns=4);
                csim            = pywt.waverec2(array2pywt( cswim ), wavelet='haar', mode='periodic');
                recovered[n,k,j] = csim
    #print(recovered)
    return recovered

def recover_vel(recovered, venc):
    #not sure if this is correct
    mag = recovered[:, 0, :]
    vel = np.empty((recovered.shape[0],) + (3,) + recovered.shape[2:] )
    for n in range(0,recovered.shape[0]):
        for k in range(1,4):
            for j in range(0, recovered.shape[2]):
                v = fft.fft2(recovered[n,k,j])
                m = fft.fft2(mag[n,j])
                v = venc/(2*math.pi)*np.log(np.divide(v,m)).imag
                vel[n,k-1,j] = v
    return np.concatenate((np.expand_dims(mag, axis=1),vel), axis=1)

if __name__ == '__main__':
    dir = home + '/Documents/undersampled/npy/'
    fourier_file = dir + 'undersampled_p50bernoulli.npy'
    undersample_file = dir + 'undersamplpattern_p50bernoulli.npy'
    pattern = np.load(undersample_file)
    #params = [(dir + 'noisy_noise5_p50.npy', pattern), (dir + 'noisy_noise10_p50.npy', pattern)]
    #for p in params:
    #    Process(target=recover, args=p).start()
    recovered = recover(fourier_file, pattern)
    imgs = recover_vel(recovered, np.load(dir + 'venc.npy'))

    print('mse between original and recovered images: ', (np.square(imgs - orig)).mean())
