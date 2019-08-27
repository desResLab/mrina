import vtk
from vtk.util.numpy_support import vtk_to_numpy
from vtk.util.numpy_support import numpy_to_vtk
import os
import numpy as np
import cv2
home = os.getenv('HOME')

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

def getInputData(directory, vtkfile):
    reader = vtk.vtkRectilinearGridReader()#vtk.vtkStructuredPointsReader()
    reader.SetFileName(directory + vtkfile)
    reader.ReadAllVectorsOn()
    reader.ReadAllScalarsOn()
    reader.Update()
    velocity, concentration = getInput(reader,directory + vtkfile)
    data = np.concatenate((concentration,velocity),axis=0)
    return data

def saveImage(fromdir, vtkfile, todir, imagefile, sliceIndex, ext='.png', numpy=False):
    inp = getInputData(directory, vtkfile)
    print(inp.shape)
    #inp = np.moveaxis(inp, 1+sliceIndex, 1)
    print(inp.shape)
    if numpy:
        np.save(todir + imagefile + '.npy', inp)
    else:
        for k in range(4):
            cv2.imwrite(todir + imagefile + '_' + str(k) + ext, inp[k]) 

if __name__ == '__main__':
    #directory = home + '/apps/undersampled/vtk/'
    directory = home + "/apps/pDat/samp256/"
    tosavedir = home + '/apps/undersampled/poiseuille/img/'
    saveImage(directory, 'pout0_0.vtk', tosavedir, 'true', sliceIndex=2)
