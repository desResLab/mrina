import vtk
from vtk.util.numpy_support import vtk_to_numpy
from vtk.util.numpy_support import numpy_to_vtk
import os
import sys
import numpy as np
import argparse
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

def saveInput(fromdir,vtkfile,todir,imagefile,sliceIndex,printlevel):
    inp = getInputData(fromdir, vtkfile)
    if(printlevel>0):
      print(inp.shape)
    inp = np.moveaxis(inp, 1+sliceIndex, 1)
    np.save(todir + imagefile + '.npy', inp)
    return inp

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Convert VTK rectilinear grid to compressed image.')
    
    parser.add_argument('-o', '--origindir',
                        action=None,
                        # nargs='+',
                        const=None,
                        default='/apps/pDat/samp256/',
                        type=str,
                        choices=None,
                        required=False,
                        help='origin VTK file folder',
                        metavar='',
                        dest='directory')

    parser.add_argument('-d', '--destdir',
                        action=None,
                        # nargs='+',
                        const=None,
                        default='/apps/undersampled/poiseuille/img/',
                        type=str,
                        choices=None,
                        required=False,
                        help='destination image folder',
                        metavar='',
                        dest='tosavedir')

    parser.add_argument('-s', '--sliceindex',
                        action=None,
                        # nargs='+',
                        const=None,
                        default=2,
                        type=str,
                        choices=None,
                        required=False,
                        help='slice index',
                        metavar='',
                        dest='sliceindex')

    parser.add_argument('-v', '--vtkname',
                        action=None,
                        # nargs='+',
                        const=None,
                        default='pout0_0.vtk',
                        type=str,
                        choices=None,
                        required=False,
                        help='name of the origin VTK file',
                        metavar='',
                        dest='vtkname')

    parser.add_argument('-f', '--imagename',
                        action=None,
                        # nargs='+',
                        const=None,
                        default='imgs_n1.npy',
                        type=str,
                        choices=None,
                        required=False,
                        help='name of the destination image file',
                        metavar='',
                        dest='imagename')

    parser.add_argument('-p', '--printlevel',
                        action=None,
                        # nargs='+',
                        const=None,
                        default=0,
                        type=str,
                        choices=None,
                        required=False,
                        help='print level, 0 - no print, >0 increasingly more information ',
                        metavar='',
                        dest='printlevel')

    # Parse Commandline Arguments
    args = parser.parse_args()

    # Convert
    saveInput(args.directory, 
              args.vtkname, 
              args.tosavedir, 
              args.imagename, 
              args.sliceindex,
              args.printlevel)

    if(args.printlevel > 0):
      print('Completed!!!')
