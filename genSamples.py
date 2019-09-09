import sys
import os
import math
import numpy as np
import numpy.fft as fft
import numpy.linalg as la
from CSRecoverySuite import generateSamplingMask, crop
home = os.getenv('HOME')
#home = 'C:/Users/Lauren/'

def getVenc(vel):
    mx = np.amax(np.fabs(vel))
    #less than 1/2: let it be equal to 1/3
    venc = mx*3
    return venc

def getKspace(sample, venc, sliceIndex=0):
    #return the fft of the complex images
    numSamples = sample.shape[0]
    mag_samples = np.empty((0,) + sample.shape[2:])
    vel_samples = np.empty((0,3,) + sample.shape[2:])
    for n in range(numSamples):
        magnitude = sample[n,0]
        y0 = np.zeros(magnitude.shape,dtype=complex)
        vel = sample[n,1:4]
        yk = np.zeros(vel.shape,dtype=complex) #reference complex images for computing velocities
        #2d fourier transform each slice for k space
        for k in range(magnitude.shape[0]): #number of 2d images in grid
            y0[k] = fft.fft2(magnitude[k])
            for j in range(0,3):
                yk[j,k] = fft.fft2(np.multiply(magnitude[k], np.exp((2*math.pi*1j/venc)*vel[j,k])))
        mag_samples = np.append(mag_samples, np.expand_dims(y0,axis=0), axis=0)
        vel_samples = np.append(vel_samples, np.expand_dims(yk,axis=0), axis=0)
    kspace = np.concatenate((np.expand_dims(mag_samples, axis=1), vel_samples), axis=1)
    return kspace

def undersample(kspace, mask):
    mag,vel = kspace
    for k in range(mag.shape[0]): #undersample differently for each sample
        mag[k,:,mask[k]] = 0
        vel[k,:,:,mask[k]] = 0
    return mag, vel

def get_noise(imsz,nrm, noise_percent,num_realizations):
    #noise param is a percentage of how much noise to be added
    noise = np.zeros((num_realizations, 4, 1,) + imsz, dtype=complex)
    snr = np.zeros((num_realizations,4))
    avgnorm = nrm[0]/math.sqrt(np.prod(imsz))
    stdev = noise * avgnorm
    for n in range(num_realizations):
        for j in range(0,4):
            avgnorm = nrm[j]/math.sqrt(np.prod(imsz))
            stdev = noise_percent * avgnorm
            snr[n,j] = math.pow(avgnorm/stdev,2)
            noise[n,j] = np.random.normal(scale=stdev, size=imsz) + 1j*np.random.normal(scale=stdev, size=imsz)
    return noise,snr

def add_noise(kspace, noise_percent, num_realizations):
    imsz = kspace.shape[3:]
    samples = np.zeros((num_realizations, 4,1,) + imsz, dtype=complex)
    nrm = [None]*4
    for v in range(4):
        nrm[v] = la.norm(kspace[0,v])
    noise, snr = get_noise(imsz, nrm, noise_percent, num_realizations)
    for n in range(num_realizations):
        for v in range(4):
            samples[n,v,0] = kspace[0,v,0] + noise[n,v]
    return samples, snr

def samples(fromdir,numRealizations,truefile='imgs_n1', tosavedir=None, numSamples=1, uType='bernoulli',genNoise=False):
    #p: percent not sampled, uType: bernoulli, poisson, halton, sliceIndex= 0,1,2 where to slice in grid
    #npydir: where to store numpy files, directory: where to retrieve vtk files, numSamples: # files to create
    if tosavedir == None:
        tosavedir = fromdir
    #get images
    inp = np.load(tosavedir + truefile + '.npy')
    print('input',inp.shape)
    venc = getVenc(inp[:,1:,:])
    np.save(tosavedir + 'venc_n' + str(numSamples) + '.npy', venc)
    kspace = getKspace(inp, venc)
    print(kspace.shape)

    for p in [0.25, 0.5, 0.75]:
        print('undersampling', p)
        mask=generateSamplingMask(1, kspace.shape[3:], p, uType)
        print(mask.shape)
        undfile = tosavedir + 'undersamplpattern_p' + str(int(p*100)) + uType + '_n' + str(numRealizations)       
        np.save(undfile, mask)
    
    if genNoise or (not os.path.exists(tosavedir + 'noisy_noise1_n' + str(numRealizations) + '.npy')):
        for noisePercent in [0.01, 0.05, 0.10, 0.30]:
            print('percent',noisePercent)
            noisy,snr = add_noise(kspace,noisePercent, numRealizations)
            fourier_file = tosavedir + 'noisy_noise' + str(int(noisePercent*100)) 
            np.save(fourier_file + '_n' + str(numRealizations), noisy)
            np.save(tosavedir + 'snr_noise' + str(int(noisePercent*100)) + '_n' + str(numRealizations), snr)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        numsamples = int(sys.argv[1])
        samptype = sys.argv[2]
        directory = sys.argv[3] #options: bernoulli, vardengauss, bpoisson, halton, vardentri, vardenexp
    else:
        numsamples = 100
        samptype = 'vardengauss'
        directory = home + "/apps/undersampled/poiseuille/img/"
    samples(directory, numsamples, uType=samptype)
