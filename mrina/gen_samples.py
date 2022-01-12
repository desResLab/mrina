import sys
import os
import math
import numpy as np
import numpy.fft as fft
import numpy.linalg as la
from mrina.mri_utils import generateSamplingMask, crop
import argparse
# import matplotlib.pyplot as plt

def getVenc(vel):
    mx = np.amax(np.fabs(vel))
    # less than 1/2: let it be equal to 1/3
    # Shouldn'd be max*1.5????
    # CHECK THIS !!!
    venc = mx*1.5
    # venc = mx*3.0
    return venc

def getKspace(sample, venc, magnitudes=None, referencephase=None):
    #return the fft of the complex images
    num_samples = sample.shape[0]
    yk = np.zeros(sample.shape,dtype=complex) #reference complex images for computing velocities
    if referencephase is None:
        print("Reference phase not found. Information loss may occur.")
        referencephase = np.zeros(sample[:,0].shape)
    if magnitudes is None:
        print("Magnitude of complex images not found. Information loss may occur.")
        magnitudes = np.ones(sample.shape)
    for n in range(num_samples):
        mag = sample[n,0]
        refphase = referencephase[n]
        vel = sample[n,1:4]
        compyk = np.zeros(vel.shape, dtype=complex)
        #2d fourier transform each slice for k space
        for k in range(sample.shape[2]): #number of 2d images in grid
            yk[n,0,k] = fft.fft2(mag[k]*np.exp(1j*refphase[k]),norm='ortho')
            for j in range(0,sample.shape[1]-1): #number of velocity components
                compyk[j,k] = magnitudes[n,j+1,k]*np.exp(1j*refphase[k])*np.exp(np.pi*1j*vel[j,k]/venc)
                yk[n,j+1,k] = fft.fft2(compyk[j,k],norm='ortho')
    return yk 

def undersample(kspace, mask):
    mag,vel = kspace
    for k in range(mag.shape[0]): #undersample differently for each sample
        mag[k,:,mask[k]] = 0
        vel[k,:,:,mask[k]] = 0
    return mag, vel

def get_noise(imsz, nrm, noise_percent, num_realizations, num_components=4):
    #noise param is a percentage of how much noise to be added
    noise = np.zeros((num_realizations, num_components, 1,) + imsz, dtype=complex)
    snr = np.zeros((num_realizations,num_components))

    for n in range(num_realizations):
        for j in range(num_components):
            # Average norm
            avgnorm = nrm[j]/math.sqrt(np.prod(imsz))
            stdev = noise_percent * avgnorm
            if(stdev < 1.0e-12):           
              # There is no noise, so SNR is infinity
              snr[n,j] = np.infty
            else:
              snr[n,j] = (avgnorm/stdev)**2
            noise[n,j] = np.random.normal(scale=stdev, size=imsz) + 1j*np.random.normal(scale=stdev, size=imsz)
    return noise,snr

def add_noise(kspace, noise_percent, num_realizations, num_components=4):
    imsz = kspace.shape[3:]
    samples = np.zeros((num_realizations, num_components, 1,) + imsz, dtype=complex)
    nrm = [None]*num_components
    for v in range(kspace.shape[1]):
        nrm[v] = la.norm(kspace[0,v])
    print('K-space image component two-norms: ', nrm)
    noise, snr = get_noise(imsz, nrm, noise_percent, num_realizations , num_components)
    for n in range(num_realizations):
        for v in range(kspace.shape[1]):
            samples[n,v,0] = kspace[0,v,0] + noise[n,v]
    return samples, snr

def genSamples(fromdir,numRealizations,truefile,tosavedir,uType,uVal,uSeed,noisePercent,useMultiPatterns,printlevel):
    
    #p: percent not sampled, uType: bernoulli, vardengauss, halton, sliceIndex= 0,1,2 where to slice in grid
    #npydir: where to store numpy files, directory: where to retrieve vtk files, numSamples: # files to create
    
    # Get images
    if(printlevel>0):
        print('Loading Image...')
    inp = np.load(fromdir + truefile + '.npy')
    
    # Get velocity encoding
    venc = getVenc(inp[:,1:,:])
    
    # Save Velocity Encoding
    if(printlevel>0):
        print('Saving velocity encoding...')    
    # Always one velocity encoding for image set
    np.save(tosavedir + 'venc_n1.npy', venc) 

    # Transform image in the Fourier domain
    if(printlevel>0):
        print('Fourier transform in k-space...')    
    kspace = getKspace(inp, venc)
    
    # Generate undersampling mask
    if(printlevel>0):
        print('Generate and save sampling mask...')  
    
    undfile = tosavedir + 'undersamplpattern_p' + str(int(uVal*100)) + uType # + '_seed' + str(uSeed)       
    if useMultiPatterns:
      numPatterns = numRealizations
      undfile = undfile + "_n" + str(numPatterns)
    else:  
      numPatterns = 1

    mask = generateSamplingMask(kspace.shape[3:], uVal, saType=uType, num_patterns=numPatterns, seed=uSeed)
    np.save(undfile, mask)

    # Add noise to image
    if(printlevel>0):
        print('Add noise to image...')
    noisy,snr = add_noise(kspace, noisePercent, numRealizations)
    fourier_file = tosavedir + 'noisy_noise' + str(int(noisePercent*100)) + '_n' + str(numRealizations)
    np.save(fourier_file, noisy)
    
    # Save SNR
    if(printlevel>0):
        print('Save image SNR...')
    np.save(tosavedir + 'snr_noise' + str(int(noisePercent*100)) + '_n' + str(numRealizations), snr)

# --- MAIN FUNCTION
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generate noisy k-space measurements and undersampling masks.')
    # fromdir         = '\.'
    parser.add_argument('-f', '--fromdir',
                        action=None,
                        # nargs='+',
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
                        # nargs='+',
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
                        # nargs='+',
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
                        # nargs='+',
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
                        # nargs='+',
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
                        # nargs='+',
                        const=None,
                        default=0.75,
                        type=float,
                        choices=None,
                        required=False,
                        help='undersampling rate',
                        metavar='',
                        dest='uVal')

    # uSeed           = 1234,
    parser.add_argument('-seed',
                        action=None,
                        # nargs='+',
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
                        # nargs='+',
                        const=None,
                        default=0.0,
                        type=float,
                        choices=None,
                        required=False,
                        help='noise percent based the average two-norm of the k-space image',
                        metavar='',
                        dest='noisepercent')
    
    # useMultiPatterns = False
    parser.add_argument('-um', '--usemultipatterns',
                        action='store_true',
                        default=False,
                        required=False,
                        help='generate a unique undersamp. pattern for each noise realization',
                        dest='usemultipatterns')

    # printlevel = 0
    parser.add_argument('-p', '--printlevel',
                        action=None,
                        # nargs='+',
                        const=None,
                        default=0,
                        type=float,
                        choices=None,
                        required=False,
                        help='print level, 0 - no print, >0 increasingly more information ',
                        metavar='',
                        dest='printlevel')

  
    # Parse Commandline Arguments
    args = parser.parse_args()
    
    # Generate Samples
    genSamples(args.fromdir,
               args.repetitions,
               args.origin,
               args.dest,
               args.uType,
               args.uVal,
               args.seed,
               args.noisepercent,
               args.usemultipatterns,
               args.printlevel)

    if(args.printlevel > 0):
      print('Completed!!!')
