import numpy as np
import matplotlib.pyplot as plt
import pydicom
import vtk
import vtk.util.numpy_support as numpy_support

def save_images(vx_folder,save_str,file_name='IM_0001'):

  # Load DICOM
  print('Reading DICOM for '+str(save_str)+'...')
  print('Reading file: ',vx_folder+file_name)
  dicom = pydicom.dcmread(vx_folder + file_name)

  # This is the pixel array which is a numpy.ndarray
  # The first index is the frame
  # The second and third indices are the pixel positions
  img = dicom.pixel_array

  # To access the scaling factors we need to access the PrivateDataTag in 
  # the DICOM file. This metadata contains the RescaleSlope and RescaleIntercept
  # parameters to rescale the values of the images.
  slope = np.array([ _metadata[0x2005,0x0140f][0].RescaleSlope for _metadata in dicom[0x5200, 0x9230] ], dtype=float)
  intercept = np.array([ _metadata[0x2005,0x0140f][0].RescaleIntercept for _metadata in dicom[0x5200, 0x9230] ], dtype=float)

  # We can also obtain resolution parameters such as pixel size (mm) and slice thickness (mm)
  PixelSpacing = np.array([ _metadata[0x2005,0x0140f][0].PixelSpacing for _metadata in dicom[0x5200, 0x9230] ], dtype=float)
  SliceThickness = np.array([ _metadata[0x2005,0x0140f][0].SliceThickness for _metadata in dicom[0x5200, 0x9230] ], dtype=float)
  SpacingBetweenSlices = np.array([ _metadata[0x2005,0x0140f][0].SpacingBetweenSlices for _metadata in dicom[0x5200, 0x9230] ], dtype=float)

  # To access the velocity encoding in cm/s we need to access the [PCVelocity] 
  # parameter in the DICOM file
  VelocityEncoding = dicom[0x2001, 0x101a].value[1]

  print('')
  print('--- Info')
  print('Pixel spacing: ',PixelSpacing[0])
  print('Slice thickness: ',SliceThickness[0])
  print('Spacing between slices: ',SpacingBetweenSlices[0])
  print('Velocity encoding: ',VelocityEncoding)
  print('')
  
  # In this case the sequence is 5240 x 192 x 192
  # The first block of 2620 frames are the density while the second block
  # with the remaining 2620 frames are the phase images
  # Each block comprises 20 time steps and 131 slices ordered by time first, 
  # e.g., the time steps for the first slice are the first 20 frames. 
  num_steps = 20
  num_slices = (img.shape[0] // 2) // num_steps
  total_slices = num_steps * num_slices * 2

  # Images are identified by (0-density,1-vx,2-vy,3-vz)
  print('Assembling images...')
  imgs = np.zeros((2, img.shape[1], img.shape[2], num_slices, num_steps), dtype=float)
  for I in range(num_slices):
      for J in range(num_steps):
          imgs[0, :, :, I , J] = img[num_steps * I + J, :, :]
          imgs[1, :, :, I , J] = img[total_slices // 2 + num_steps * I + J, :, :]

  # We rescale the density and phase images accordingly
  print('Scaling images...')
  for I in range(num_slices):
      for J in range(num_steps):
          imgs[0,:, :, I , J] = slope[num_steps * I + J] * imgs[0, :, :, I, J] + intercept[num_steps * I + J]
          imgs[1,:, :, I , J] = slope[total_slices // 2 + num_steps * I + J] * imgs[1, :, :, I, J] + intercept[total_slices // 2 + num_steps * I + J]

  # Save in numpy array
  np.save('img_'+str(save_str)+'.npy',imgs)

def export_vti_solid(imgs,file_name):

  data_type = vtk.VTK_FLOAT

  img = vtk.vtkImageData()
  img.SetDimensions(imgs.shape[3], imgs.shape[2], imgs.shape[1])

  # Loop over the time steps
  for loopA in range(imgs.shape[4]):

    print('Exporting frame '+str(loopA+1)+'...')

    # Get current image
    curr_img = imgs[:,:,:,:,loopA]

    # Densities
    flat_data_array = curr_img[0].flatten()
    vtk_data = numpy_support.numpy_to_vtk(num_array=flat_data_array, deep=True, array_type=data_type)
    vtk_data.SetName('density')

    img.GetPointData().SetScalars(vtk_data)

    flat_data_array = np.zeros((np.prod(curr_img[0].shape),3))
    flat_data_array[:,0] = curr_img[1].flatten()
    flat_data_array[:,1] = curr_img[2].flatten()
    flat_data_array[:,2] = curr_img[3].flatten()  
    vtk_data = numpy_support.numpy_to_vtk(num_array=flat_data_array.flatten(), deep=True, array_type=data_type)
    vtk_data.SetName('velocity')
    vtk_data.SetNumberOfComponents(3)

    img.GetPointData().SetVectors(vtk_data)
    
    # Export to VTI
    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(file_name + '_' + str(loopA).zfill(3) + '.vti')
    writer.SetInputData(img)
    writer.Write()

def get_base2(dims):
  dims_base2 = (2**np.log2(dims).astype(int))
  return dims_base2

def export_slice(imgs,file_name='imgs_n1.npy',step=5,mode='xy',slice_num=50,save_npy=True):
  
  # Get image dimensions
  if(mode == 'xy'):
    img_dims = np.array([imgs.shape[1],imgs.shape[2]])
  elif(mode == 'yz'):
    img_dims = np.array([imgs.shape[2],imgs.shape[3]])
  elif(mode == 'xz'):
    img_dims = np.array([imgs.shape[1],imgs.shape[3]])
  else:
    print('Invalid slice mode. Valid modes are xy, yz and xz.')
    exit()

  # Extract base 2 sizes 
  dims_base2  = get_base2(img_dims)
  # Determine start and end of slice
  ss = (img_dims-dims_base2)//2 # start slice
  es = (ss + dims_base2) # end slice

  # Allocate new vector
  img_to_export = np.zeros((1,4,1,dims_base2[0],dims_base2[1]))
  
  if(mode == 'xy'):
    tmp = imgs[:,ss[0]:es[0],ss[1]:es[1],slice_num,step]
    # Assign density and velocity components
    img_to_export[0,0,0] =  tmp[0]
    img_to_export[0,1,0] =  tmp[2]
    img_to_export[0,2,0] = -tmp[3]
    img_to_export[0,3,0] = -tmp[1]
  elif(mode == 'yz'):
    tmp = imgs[:,slice_num,ss[0]:es[0],ss[1]:es[1],step]
    # Assign density and velocity components
    img_to_export[0,0,0] =  tmp[0]
    img_to_export[0,1,0] =  tmp[1]
    img_to_export[0,2,0] = -tmp[2]
    img_to_export[0,3,0] = -tmp[3]
  elif(mode == 'xz'):
    tmp = imgs[:,ss[0]:es[0],slice_num,ss[1]:es[1],step]
    # Assign density and velocity components
    img_to_export[0,0,0] =  tmp[0]
    img_to_export[0,1,0] =  tmp[1]
    img_to_export[0,2,0] = -tmp[3]
    img_to_export[0,3,0] =  tmp[2]
  else:
    print('Invalid slice mode. Valid modes are xy, yz and xz.')
    exit()

  # return image
  if(save_npy):
    np.save(file_name,img_to_export)

  return img_to_export

def plot_slice(sl,slice_id='01'):
  plt.figure(figsize=(10,3))
  plt.subplot(1,4,1)
  plt.imshow(sl[0,0,0])
  plt.axis(False)
  plt.subplot(1,4,2)
  plt.imshow(sl[0,1,0])
  plt.axis(False)
  plt.subplot(1,4,3)
  plt.imshow(sl[0,2,0])
  plt.axis(False)
  plt.subplot(1,4,4)
  plt.imshow(sl[0,3,0])
  plt.axis(False)
  plt.show()
  plt.close()

  plt.figure()
  plt.imshow(sl[0,0,0])
  plt.axis(False)
  plt.tight_layout()
  plt.savefig('slice'+str(slice_id)+'.png',bbox_inches='tight', pad_inches=0)
  plt.close()

# ====
# MAIN
# ====
if __name__ == "__main__":

  export_vti = False

  if(False):
    save_images('../01_data/HRFFE/AP/','AP')
    save_images('../01_data/HRFFE/FH/','FH')
    save_images('../01_data/HRFFE/RL/','RL')
    exit()
  else:
    img_ap = np.load('img_AP.npy')
    img_fh = np.load('img_FH.npy')
    img_rl = np.load('img_RL.npy')
    
    # Assemble final image
    img = np.zeros((4,)+img_ap.shape[1:])
    img[0] =  img_ap[0]
    img[1] = -img_rl[1] # x velocity
    img[2] =  img_ap[1] # y velocity
    img[3] = -img_fh[1] # z velocity

    # Export to VTI
    if(export_vti):
      export_vti_solid(img,'export')
    else:
      slice_01 = export_slice(img,file_name='imgs_n1_01.npy',step=5,mode='xy',slice_num=50)
      slice_02 = export_slice(img,file_name='imgs_n1_02.npy',step=5,mode='yz',slice_num=100)
      slice_03 = export_slice(img,file_name='imgs_n1_03.npy',step=5,mode='xz',slice_num=122)

      plot_slice(slice_01,'01')
      plot_slice(slice_02,'02')
      plot_slice(slice_03,'03')

