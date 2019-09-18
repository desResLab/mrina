import vtk
from vtk.util.numpy_support import vtk_to_numpy
from vtk.util.numpy_support import numpy_to_vtk
import os
import sys
import numpy as np
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

def saveInput(fromdir, vtkfile, todir, imagefile, sliceIndex, ext='.png', numpy=False):
    inp = getInputData(directory, vtkfile)
    print(inp.shape)
    inp = np.moveaxis(inp, 1+sliceIndex, 1)
    np.save(todir + imagefile + '.npy', inp)
    return inp

if __name__ == '__main__':
    if len(sys.argv) > 1:
        directory = sys.argv[1]
        tosavedir = sys.argv[2]
        sliceIndex = int(sys.argv[3])
    else: 
        directory = home + "/apps/pDat/samp256/"
        tosavedir = home + '/apps/undersampled/poiseuille/img/'
        sliceIndex = 2

    saveInput(directory, 'pout0_0.vtk', tosavedir, 'imgs_n1', sliceIndex=sliceIndex)
