import sys
import os
import math
import numpy as np
import numpy.fft as fft
import numpy.linalg as la
from CSRecoverySuite import generateSamplingMask, crop
import argparse

home = os.getenv('HOME')

def getVenc(vel):
    mx = np.amax(np.fabs(vel))
    #less than 1/2: let it be equal to 1/3
    # Shouldn'd be mx*1.5????
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

def genSamples(fromdir,numRealizations,truefile,tosavedir,uType,uVal,uSeed,noisePercent):
    
    #p: percent not sampled, uType: bernoulli, vardengauss, halton, sliceIndex= 0,1,2 where to slice in grid
    #npydir: where to store numpy files, directory: where to retrieve vtk files, numSamples: # files to create
    
    # Get images
    inp = np.load(fromdir + truefile + '.npy')
    
    # Get velocity encoding
    venc = getVenc(inp[:,1:,:])
    
    # Save Velocity Encoding
    np.save(tosavedir + 'venc_n' + str(numSamples) + '.npy', venc)

    # Transform image in the Fourier domain
    kspace = getKspace(inp, venc)
    
    # Generate undersampling mask
    mask = generateSamplingMask(kspace.shape[3:], uVal, uType, uSeed)
    undfile = tosavedir + 'undersamplpattern_p' + str(int(uVal*100)) + uType + '_seed' + str(uSeed)       
    np.save(undfile, mask)
    
    noisy,snr = add_noise(kspace, noisePercent, numRealizations)
    fourier_file = tosavedir + 'noisy_noise' + str(int(noisePercent*100)) 
    np.save(fourier_file + '_n' + str(numRealizations), noisy)
    
    # Save SNR
    np.save(tosavedir + 'snr_noise' + str(int(noisePercent*100)) + '_n' + str(numRealizations), snr)

# --- MAIN FUNCTION
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generate noisy k-space measurements and undersampling masks.')
    # fromdir         = '\.'
    parser.add_argument('-f', '--fromdir',
                        action=None,
                        nargs=1,
                        const=None,
                        default='./',
                        type=str,
                        choices=None,
                        required=False,
                        help='origin image location',
                        metavar='',
                        dest='fromdir')

    # numRealizations = 1
    parser.add_argument('-r', '--repetitions',
                        action=None,
                        nargs=1,
                        const=None,
                        default=1,
                        type=int,
                        choices=None,
                        required=False,
                        help='number of k-space samples and mask to generate',
                        metavar='',
                        dest='repetitions')

    # truefile        = 'imgs_n1', 
    parser.add_argument('-o', '--origin',
                        action=None,
                        nargs=1,
                        const=None,
                        default='imgs_n1',
                        type=str,
                        choices=None,
                        required=False,
                        help='name of the origin images to process',
                        metavar='',
                        dest='origin')

    # tosavedir       = None, 
    parser.add_argument('-d', '--dest',
                        action=None,
                        nargs=1,
                        const=None,
                        default='./',
                        type=str,
                        choices=None,
                        required=False,
                        help='destination folder for sample generation',
                        metavar='',
                        dest='dest')

    # uType           = 'bernoulli',
    parser.add_argument('-u', '--utype',
                        action=None,
                        nargs=1,
                        const=None,
                        default='bernoulli',
                        type=str,
                        choices=['bernoulli','vardentri','vardengauss','vardenexp','halton'],
                        required=False,
                        help='undersampling pattern type',
                        metavar='',
                        dest='uType')

    # uVal            = 0.5,
    parser.add_argument('-v', '--urate',
                        action=None,
                        nargs=1,
                        const=None,
                        default=True,
                        type=float,
                        choices=None,
                        required=False,
                        help='undersampling rate',
                        metavar='',
                        dest='uVal')

    # uSeed           = 1234,
    parser.add_argument('-seed',
                        action=None,
                        nargs=1,
                        const=None,
                        default=1234,
                        type=int,
                        choices=None,
                        required=False,
                        help='random generator seed',
                        metavar='',
                        dest='seed')

    # noisePercent    = 0.0
    parser.add_argument('-n', '--noisepercent',
                        action=None,
                        nargs=1,
                        const=None,
                        default=0.0,
                        type=float,
                        choices=None,
                        required=False,
                        help='noise ',
                        metavar='',
                        dest='noisepercent')

    args = parser.parse_args()
    
    # Generate Samples
    genSamples(args.fromdir,
               args.repetitions,
               args.origin,
               args.dest,
               args.uType,
               args.uVal,
               args.seed,
               args.noisepercent)
