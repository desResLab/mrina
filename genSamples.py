import sys
import os
import math
import pydoc
import vtk
from vtk.util.numpy_support import vtk_to_numpy
from vtk.util.numpy_support import numpy_to_vtk
from scipy.stats import bernoulli
import numpy as np
import numpy.fft as fft
import numpy.linalg as la
import sigpy.mri as mri

home = os.getenv('HOME')
#home = 'C:/Users/Lauren/'

def next_prime():
    def is_prime(num):
    #"Checks if num is a prime value"
        for i in range(2,int(num**0.5)+1):
            if(num % i)==0: return False
            return True
    prime = 3
    while(1):
        if is_prime(prime):
            yield prime
        prime += 2
def vdc(n, base=2):
    vdc, denom = 0, 1
    while n:
        denom *= base
        n, remainder = divmod(n, base)
        vdc += remainder/float(denom)
    return vdc
def halton_sequence(size, dim):
    seq = []
    primeGen = next_prime()
    next(primeGen)
    for d in range(dim):
        base = next(primeGen)
        seq.append([vdc(i, base) for i in range(size)])
    return seq

def getInput(reader, filename):
    reader.SetFileName(filename)
    reader.Update()
    polyDataModel = reader.GetOutput()
    dims = polyDataModel.GetDimensions()
    data = polyDataModel.GetPointData()
    velocity = vtk_to_numpy(data.GetArray('velocity'))
    velocity = np.reshape(velocity, (dims[2], dims[1], dims[0],3))
    velocity = np.transpose(velocity, (3,2,1,0))
    concentration = vtk_to_numpy(data.GetScalars('concentration'))
    concentration = np.reshape(concentration, (dims[2], dims[1], dims[0],1))
    concentration = np.transpose(concentration, (3,2,1,0))
    return velocity,concentration

def getInputData(directory, nums):
    reader = vtk.vtkRectilinearGridReader()#vtk.vtkStructuredPointsReader()
    reader.SetFileName(directory + "pout0_0.vtk")
    reader.ReadAllVectorsOn()
    reader.ReadAllScalarsOn()
    reader.Update()
    velocity, concentration = getInput(reader,directory + "pout0_0.vtk")
    data = np.concatenate((concentration,velocity),axis=0)
    samples = np.empty((0,) + data.shape)
    print(samples.shape)
    for k in nums:
        filename = directory + "pout" + str(k) + "_0.vtk"
        velocity, concentration = getInput(reader,filename)
        data = np.concatenate((concentration,velocity),axis=0)
        samples =np.append(samples,np.expand_dims(data,axis=0), axis=0)
    return samples

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
    return mag_samples, vel_samples

def undersamplingMask(shape,p,type='bernoulli'):
    mask = np.empty((shape[0], shape[2],shape[3]), dtype=np.bool)
    for k in range(shape[0]):
        #to keep undersampling the same for each slice
        if type=='bernoulli':
            indices = bernoulli.rvs(size=(shape[2], shape[3]), p=p)
        elif type =='poisson': #poisson
            accel =  1/p  #accel: Target acceleration factor. Greater than 1.
            indices = mri.poisson((shape[2],shape[3]), accel)
        else: #halton sequence
            numPts = int(p*shape[2]*shape[3])
            pts = np.transpose(np.asarray(halton_sequence(numPts, 2)))
            pts[:,0] = pts[:,0]*shape[2]
            pts[:,1] = pts[:,1]*shape[3]
            pts = pts.astype(int)
            indices = np.zeros((shape[2],shape[3]))
            indices[pts[:,0], pts[:,1]] = 1
        mask[k] = np.ma.make_mask(indices)
    return mask

def undersample(kspace, mask):
    mag,vel = kspace
    for k in range(mag.shape[0]): #undersample differently for each sample
        mag[k,:,mask[k]] = 0
        vel[k,:,:,mask[k]] = 0
    return mag, vel

def add_noise(kspace,noise):
    #noise param is a percentage of how much noise to be added
    mag,vel=kspace
    snr = np.zeros((mag.shape[0],4))
    for k in range(0, mag.shape[0]):
        avgnorm = la.norm(mag[k,:])/math.sqrt(mag[k,:].size)
        stdev = noise * avgnorm
        snr[k,0] = math.pow(avgnorm/stdev,2)
        mag[k,:] = mag[k,:] + np.random.normal(scale=stdev, size=mag.shape[1:]) + 1j*np.random.normal(scale=stdev, size=mag.shape[1:])
        for j in range(0,3):
            avgnorm = la.norm(vel[k,j,:])/math.sqrt(vel[k,j,:].size)
            stdev = noise * avgnorm
            snr[k,j+1] = math.pow(avgnorm/stdev,2)
            vel[k,j,:] = vel[k,j,:] + np.random.normal(scale=stdev, size=vel.shape[2:]) + 1j*np.random.normal(scale=stdev, size=vel.shape[2:])
    return (mag,vel),snr

def samples(directory, numSamples, p=0.5, uType='bernoulli', sliceIndex=0,npydir=home +'/Documents/undersampled/npy/'):
    #p: percent not sampled, uType: bernoulli, poisson, halton, sliceIndex= 0,1,2 where to slice in grid
    #npydir: where to store numpy files, directory: where to retrieve vtk files, numSamples: # files to create
    print(range(0,numSamples))
    inp = getInputData(directory, range(0,numSamples))
    inp = np.moveaxis(inp, 2+sliceIndex, 2)
    np.save(npydir + 'imgs.npy', inp)
    venc = getVenc(inp[:,1:,:])
    np.save(npydir + 'venc.npy', venc)
    kspace = getKspace(inp, venc)
    mask=undersamplingMask(kspace[0].shape,p,uType)

    undersampled = undersample(kspace,mask)
    undersampled = np.concatenate((np.expand_dims(undersampled[0], axis=1), undersampled[1]), axis=1)
    np.save(npydir + 'undersampled_p' + str(int(p*100)) + uType, undersampled)
    np.save(npydir + 'undersamplpattern_p' + str(int(p*100)) + uType, mask)

    for noisePercent in [0.05, 0.10, 0.30]:
        print('percent',noisePercent)
        print(kspace[0].shape, kspace[1].shape)
        noisy,snr = add_noise(kspace,noisePercent)
        #noisy = undersample(noisy,mask)
        noisy = np.concatenate((np.expand_dims(noisy[0], axis=1), noisy[1]), axis=1)
        np.save(npydir + 'noisy_noise' + str(int(noisePercent*100)) + '_p' + str(int(p*100)) + uType, noisy)
        np.save(npydir + 'snr_noise' + str(int(noisePercent*100)) + '_p' + str(int(p*100)) + uType, snr)

if __name__ == '__main__':

    #directory = home + '/apps/undersampled/vtk/'
	#directory = home + "/Documents/undersampled/vtk/"
    directory = home + "/Documents/poiseuille/256/"
    samples(directory, 1, p=0.1, uType='bernoulli',sliceIndex=2,npydir=home+"/Documents/undersampled/poiseuille/npy/")
