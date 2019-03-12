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
#sys.path.insert('../unet/unet/')
#import unet as u

def getInput(reader, filename):
	reader.SetFileName(filename)
	reader.Update()
	polyDataModel = reader.GetOutput()
	dim = polyDataModel.GetDimensions()
	data = polyDataModel.GetPointData()
	velocity = vtk_to_numpy(data.GetArray('velocity'))
	velocity = np.reshape(velocity, (3,) + dim)
	concentration = vtk_to_numpy(data.GetScalars('concentration'))
	concentration = np.reshape(concentration, (1,) + dim)
	return velocity,concentration

def getInputData(directory, data):
	reader = vtk.vtkStructuredPointsReader()
	reader.SetFileName(directory + "pout0_0.vtk")
	reader.ReadAllVectorsOn()
	reader.ReadAllScalarsOn()
	reader.Update()
	samples = np.empty((0,4,16,16,16))
	for k in data:
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

def getKspace(sample, venc):
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
        else: #poisson
            accel =  1/p  #accel: Target acceleration factor. Greater than 1.
            indices = mri.poisson((shape[2],shape[3]), accel)
        mask[k] = np.ma.make_mask(indices)
    return mask

def undersample(kspace, mask):
    mag,vel = kspace
    for k in range(mag.shape[0]): #undersample differently for each sample
        mag[k,:,mask[k]] = 0
        vel[k,:,:,mask[k]] = 0
    return mag, vel

def add_noise(kspace,noise):
    #noise is a percentage of how much noise to be added
    mag,vel=kspace
    print('add noise')
    #print(mag)
    snr = np.zeros((mag.shape[0],4))
    for k in range(0, mag.shape[0]):
        #print(mag[k,:].shape)
        avgnorm = la.norm(mag[k,:])/math.sqrt(mag[k,:].size)
        stdev = noise * avgnorm
        #print(avgnorm, stdev, la.norm(vel[k,0]))
        snr[k,0] = math.pow(avgnorm/stdev,2)
        mag[k,:] = mag[k,:] + np.random.normal(scale=stdev, size=mag.shape[1:]) + 1j*np.random.normal(scale=stdev, size=mag.shape[1:])
        for j in range(0,3):
            avgnorm = la.norm(vel[k,j,:])/math.sqrt(vel[k,j,:].size)
            #print(avgnorm,stdev, la.norm(vel[k,j,0]))
            stdev = noise * avgnorm
            snr[k,j+1] = math.pow(avgnorm/stdev,2)
            vel[k,j,:] = vel[k,j,:] + np.random.normal(scale=stdev, size=vel.shape[2:]) + 1j*np.random.normal(scale=stdev, size=vel.shape[2:])
    return (mag,vel),snr

def samples(directory, numSamples, p=0.5, uType='bernoulli', npydir=home +'/Documents/undersampled/npy/'):
    print(range(0,numSamples))
    inp = getInputData(directory, range(0,numSamples))
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
        noisy = undersample(noisy,mask)
        noisy = np.concatenate((np.expand_dims(noisy[0], axis=1), noisy[1]), axis=1)
        np.save(npydir + 'noisy_noise' + str(int(noisePercent*100)) + '_p' + str(int(p*100)), noisy)
        np.save(npydir + 'snr_noise' + str(int(noisePercent*100)) + '_p' + str(int(p*100)), snr)

if __name__ == '__main__':

    #directory = home + '/apps/undersampled/vtk/'
	directory = home + "/Documents/undersampled/vtk/"
	samples(directory, 1)
