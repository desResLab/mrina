import sys,os
import random
import numpy as np
import matplotlib.pyplot as plt
sys.path.append('../')
from mrina.mri_utils import extractFluidMask
from mrina.recover import recover_vel
import argparse

home = os.getenv('HOME')

# Parameters
EPSILON = 0.5

def circle(shape, center, dist, useFluidMask=False, fluidMask=None):
  # probably a more efficient way to determine this
  pts = []
  for x in range(max(0,center[0]-dist), min(center[0]+dist, shape[0])):
    for y in range(max(0,center[1]-dist), min(center[1]+dist, shape[1])):
      if abs((x-center[0])**2 + (y-center[1])**2 - dist**2) < EPSILON**2:
        if(useFluidMask):
          if(fluidMask[x,y]):
            pts.append([x, y])
        else:
          pts.append([x, y])
  return pts

def pick_random_point(imsz, useFluidMask=False, fluidMask=None):
  if(useFluidMask):
    found = False
    while(not found):      
      pt1 = (random.randint(0, imsz[0]-1), random.randint(0, imsz[1]-1))
      found = fluidMask[pt1[0],pt1[1]]
  else:
    pt1 = (random.randint(0, imsz[0]-1), random.randint(0, imsz[1]-1))
  return pt1

def select_points(dist, imsz, useFluidMask=False, fluidMask=None):
  # Select center at random
  pt1 = pick_random_point(imsz, useFluidMask, fluidMask)
  choices = circle(imsz, pt1, dist, useFluidMask, fluidMask)
  # Not sure why this loop is here...
  while not choices: # make sure there were points within range for distance
    pt1 = pick_random_point(imsz, useFluidMask, fluidMask)
    choices = circle(imsz, pt1, dist, useFluidMask, fluidMask)
  # Select pt2 at random from the list
  pt2 = tuple(random.choice(choices))
  if pt1 <= pt2: # pt1[0] < pt2[0] or (pt1[0] == pt2[0] and pt1[1] <= pt2[1]):
    return pt1,pt2
  return pt2, pt1

def get_points(size, num_pts, imsz, ptsdir, pointFileName, useFluidMask=False, fluidMask=None):
  print('Getting points of size ' + str(size) + ' and ' + str(num_pts) + ' points...')
  points = np.zeros((size, num_pts, 2, 2), dtype=int)
  for k in range(1,size+1):
    if(k % 10 == 0):
      print(k)
    for j in range(num_pts):
      pt1, pt2 = select_points(k,imsz,useFluidMask,fluidMask)
      while pt1[0] in points[k-1,:,0,0] and pt1[1] in points[k-1,:,0,1] and pt2[0] in points[k-1,:,1,0] and pt2[0] in points[k-1,:,1,1]:
        # Select new points if we've already selected these
        pt1, pt2 = select_points(k,imsz,useFluidMask,fluidMask) 
      points[k-1, j, 0, 0] = pt1[0]
      points[k-1, j, 0, 1] = pt1[1]
      points[k-1, j, 1, 0] = pt2[0]
      points[k-1, j, 1, 1] = pt2[1]
  # Save Point File
  np.save(pointFileName,points)

def get_samples(recfile, singlechannel, vencdir):
  # Load reconstructed images
  samples = np.load(recfile)
  # Get Vencoding
  vencfile = vencdir + 'venc_n1' + '.npy'
  if os.path.exists(vencfile):
    venc = np.load(vencfile)
  else:
    venc = None
  # Recover the velocities
  if(singlechannel):
    samples = np.absolute(samples)
  else:
    samples = recover_vel(samples, venc)
  # Compute average reconstructed image
  avgimg = np.mean(samples, axis=0)
  # Compute reconstruction noise and return
  for k in range(samples.shape[0]):
    samples[k] = samples[k] - avgimg    
  return samples

def get_saved_points(samples, size, num_pts, maindir, ptsdir, useFluidMask, fluidMaskFile):
  imsz = samples.shape[3:]

  # Get File Name
  fluidMask = None
  if(useFluidMask):
    if(fluidMaskFile == ''):
      print('Using fluid mask...')
    else:
      print('Using fluid mask from file: '+maindir+fluidMaskFile+'...')

    # Get File with Original Image
    orig_file = maindir + 'imgs_n1.npy'
  
    # Check if files exist and open the images
    if(os.path.exists(orig_file)):
      orig = np.load(orig_file).astype(np.complex)
    else:
      print('ERROR: Cannot find file with original image.')
      exit(-1)

    if(fluidMaskFile == ''):
      # Extract fluid mask for simple geometries
      fluidMask = extractFluidMask(orig)
    else:
      # Read mask from file 
      fluidMask = np.load(maindir + fluidMaskFile)

    if(False):
      # Plot the fluid mask for debug
      plt.imshow(fluidMask,cmap='gray')
      plt.show()
      exit()

    # Set File Name for Point in the Fluid Mask
    pointFileName = ptsdir+'points_s' + str(size) + '_n' + str(num_pts) + '_fluidmask.npy'
  else:

    # Set the file name for points without fluid mask
    pointFileName = ptsdir+'points_s' + str(size) + '_n' + str(num_pts) + '.npy'

  # Read or generate the points
  if not os.path.isfile(pointFileName):
    get_points(size, num_pts, imsz, ptsdir, pointFileName, useFluidMask, fluidMask)
  points = np.load(pointFileName)
  return points

def get_coeff(size, num_pts, samples, points):
  print('in coeff', samples.shape)
  coeff = np.zeros((samples.shape[1], size, num_pts))
  for k in range(1, size+1):
    for j in range(num_pts):
      pt1 = (points[k-1, j, 0,0], points[k-1,j, 0,1])
      pt2 = (points[k-1, j, 1,0], points[k-1,j, 1,1])
      for v in range(samples.shape[1]):
        var1 = np.abs(samples[:, v, 0, pt1[0], pt1[1]])
        var2 = np.abs(samples[:, v, 0, pt2[0], pt2[1]])
        # If both sequences contain zeros, this generates a nan.
        # We replace nan with ones. This is only a problem when evaluating the 
        # correlation between pixels in the background. 
        coeff[v, k-1, j] = np.corrcoef(np.asarray([var1, var2]))[0,1] # correlation
  if(np.isnan(coeff).any()):
    print('WARNING: Found some NAN, probably due to sequences of zeros in the background. Replacing with correlation of 1.0')
    coeff[np.isnan(coeff)] = 1.0
  return coeff

def get_vals(recfile, savefile, maxcorrpixeldist, numpts, singlechannel, maindir, recdir, ptsdir, useFluidMask, fluidMaskFile, save_numpy=True):
  samples = get_samples(recfile, singlechannel, args.vencdir)
  print('Use Fluid Mask: ',useFluidMask)
  points  = get_saved_points(samples, maxcorrpixeldist, numpts, maindir, ptsdir, useFluidMask, fluidMaskFile)
  coeff   = get_coeff(maxcorrpixeldist, numpts, samples, points)
  if save_numpy:
    np.save(savefile, coeff)
  return coeff

def get_all(args):
  # size is max. distance to retrieve correlations for
  # num_pts is the number of points to average correlation across
  for noise_percent in [0.0, 0.01, 0.05, 0.1, 0.3]:
    for p in [0.25, 0.5, 0.75, 0.80, 0.85, 0.90, 0.95]:
      for wavetype in ['HAAR','DB8']:
        for algtype in ['CS','CSDEB','OMP']:
          for samptype in ['bernoulli', 'vardengauss']:
            recfile = args.recdir + 'rec_noise' + str(int(noise_percent*100)) + \
                                    '_p' + str(int(p*100)) + samptype + \
                                    '_n' + str(args.numsamples) + \
                                    '_w' + str(wavetype) + \
                                    '_a' + str(algtype) + '.npy'

            savefile = args.recdir + 'corrcoeff' + str(args.numpts) + \
                                     '_noise' + str(int(noise_percent*100)) + \
                                     '_p' + str(int(p*100)) + samptype + \
                                     '_n'+str(args.numsamples)+ \
                                     '_w' + str(wavetype) + \
                                     '_a' + str(algtype)
            
            # Change name for fluid Mask 
            if(args.usefluidmask):
              savefile += '_fluid.npy'
            else:
              savefile += '.npy'

            if(os.path.exists(recfile)):
              get_vals(recfile, savefile, args.maxcorrpixeldist, args.numpts, args.singlechannel, args.maindir, args.recdir, args.ptsdir, args.usefluidmask, args.fluidmaskfile)
              print('Saved! Noise: ', noise_percent,', undersampling prob: ', p, ', undersampling mask: ', samptype, ', wavetype: ', wavetype, ', algorithm: ', algtype)

# MAIN 
if __name__ == '__main__':

  # Init parser
  parser = argparse.ArgumentParser(description='Generate result images.')

  # numsamples
  parser.add_argument('-n', '--numsamples',
                      action=None,
                      # nargs='+',
                      const=None,
                      default=100,
                      type=int,
                      choices=None,
                      required=False,
                      help='number of repetitions',
                      metavar='',
                      dest='numsamples')
  # numpts
  parser.add_argument('-m', '--numpts',
                      action=None,
                      # nargs='+',
                      const=None,
                      default=50,
                      type=int,
                      choices=None,
                      required=False,
                      help='number of point pairs for computing correlations',
                      metavar='',
                      dest='numpts')
  # size
  parser.add_argument('-c', '--maxcorrpixeldist',
                      action=None,
                      # nargs='+',
                      const=None,
                      default=100,
                      type=int,
                      choices=None,
                      required=False,
                      help='Maximum integer pixel distance to use for correlation computation',
                      metavar='',
                      dest='maxcorrpixeldist')
  # recdir
  parser.add_argument('-d', '--maindir',
                      action=None,
                      # nargs='+',
                      const=None,
                      default='./',
                      type=str,
                      choices=None,
                      required=False,
                      help='folder containing the original image',
                      metavar='',
                      dest='maindir')  
  # recdir
  parser.add_argument('-r', '--recdir',
                      action=None,
                      # nargs='+',
                      const=None,
                      default='./',
                      type=str,
                      choices=None,
                      required=False,
                      help='folder containing the non linear reconstructions',
                      metavar='',
                      dest='recdir')
  # ptsdir
  parser.add_argument('-t', '--ptsdir',
                      action=None,
                      # nargs='+',
                      const=None,
                      default='./',
                      type=str,
                      choices=None,
                      required=False,
                      help='folder containing the saved point coordinates',
                      metavar='',
                      dest='ptsdir')
  # vencdir
  parser.add_argument('-v', '--vencdir',
                      action=None,
                      # nargs='+',
                      const=None,
                      default='./',
                      type=str,
                      choices=None,
                      required=False,
                      help='folder containing the velocity encoding',
                      metavar='',
                      dest='vencdir')
  # singlechannel
  parser.add_argument('--singlechannel',
                      action='store_true',
                      default=False,
                      required=False,
                      help='treat the image as single-channel, without velocity components',
                      dest='singlechannel')  
  # usefluidmask
  parser.add_argument('--usefluidmask',
                      action='store_true',
                      default=False,
                      required=False,
                      help='compute correlation only within fluid region',
                      dest='usefluidmask')  

  # fluidmaskfile
  parser.add_argument('--fluidmaskfile',
                      action=None,
                      # nargs='+',
                      const=None,
                      default='',
                      type=str,
                      choices=None,
                      required=False,
                      help='name of the npy file containing the binary mask',
                      metavar='',
                      dest='fluidmaskfile')  

  # Print Level
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

  # Save all the images
  get_all(args)

  # Completed!
  if(args.printlevel > 0):
    print('Completed!!!')  
