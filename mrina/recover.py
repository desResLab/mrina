import sys,os
import numpy as np
import matplotlib.pyplot as plt
import numpy.fft as fft
import pywt
import math
import scipy.misc
from scipy.stats import norm
from multiprocessing import Process, cpu_count, Manager
from mrina.gen_samples import getKspace,getVenc
from mrina.mri_utils import crop
# Import Solvers
from mrina.maps import OperatorWaveletToFourier
from mrina.solver_l1_norm import RecoveryL1NormNoisy, MinimizeSumOfSquares
from mrina.solver_omp import OMPRecovery
import argparse

home = os.getenv('HOME')

CS_MODE     = 0
DEBIAS_MODE = 1
OMP_MODE    = 2

# OMP algorithm defaults
def_omp_mode     = 'stomp'
def_omp_iter     = 10
def_omp_ts       = 2.0
def_omp_usenorms = False

def get_eta(im, imNrm, noise_percent, m):
  avgnorm = imNrm/math.sqrt(im.size)
  stdev   = noise_percent * avgnorm
  rv      = norm()
  # ppf: inverse of cdf
  eta     = stdev*math.sqrt(2*m + 2*math.sqrt(m)*rv.ppf(0.95))
  if eta < 1E-3: # min. threshold for eta
    eta = 1E-3
  return eta

def recoverOne(kspace, imsz, eta, omega, wvlt='haar', 
               solver_mode=CS_MODE, omp_mode=def_omp_mode, omp_iter=def_omp_iter, omp_ts=def_omp_ts, omp_usenorms=def_omp_usenorms):
  
  A = OperatorWaveletToFourier(imsz, samplingSet=omega, waveletName=wvlt)
  
  wim = pywt.wavedec2(fft.ifft2(kspace, norm='ortho'), wavelet=wvlt, mode='zero')
  wim = pywt.coeffs_to_array(wim)[0]

  yim = A.eval(wim, 1)
  
  # each recovery method returns (solution, norm) 
  # here we're only interested in the solution
  if(solver_mode == OMP_MODE):
    
    # OMP Recovery
    tol = eta/np.linalg.norm(yim.ravel(),2)
    print('Recovering using OMP with tol = %8.3e' % (tol))

    wim = OMPRecovery(A, yim, tol=tol, ompMethod=omp_mode, maxItns=omp_iter, ts_factor=omp_ts, useNorms=omp_usenorms)[0]

  else:
    # CS Recovery
    print('Recovering using CS with eta =', eta)
    wim =  RecoveryL1NormNoisy(eta, yim, A, disp=True)

  if isinstance(wim, tuple):
    wim = wim[0] # for the case where ynrm is less than eta
  
  # CS-DEBIAS Recovery
  if(solver_mode == DEBIAS_MODE):
    support = (np.abs(wim.flatten()) > 1E-3)
    Adeb = A.colRestrict(basisSet=support)
    wim =  MinimizeSumOfSquares(yim, Adeb)
    if isinstance(wim, tuple):
      wim = wim[0] # for the case where ynrm is less than eta

  # Reconstruct Image from wavelet coefficients
  csim = A.getImageFromWavelet(wim)

  return csim
  
def recover(noisy, original, pattern, noise_percent, processnum, return_dict, wvlt='haar', 
            solver_mode=CS_MODE, omp_mode=def_omp_mode, omp_iter=def_omp_iter, omp_ts=def_omp_ts, omp_usenorms=def_omp_usenorms):    
  imsz = crop(noisy[0,0,0]).shape
  cs = np.zeros(noisy.shape[0:3] + imsz,dtype=complex)
  print('Pattern shape: ', pattern.shape)
  for n in range(noisy.shape[0]):
    if len(pattern.shape) > 2:
      if pattern.shape[0] == 1:
        idx = 0
      else:
        idx = n
      omega = crop(pattern[idx])
    else:
      omega = crop(pattern)
    for k in range(noisy.shape[1]):
      for j in range(noisy.shape[2]):
        im = crop(original[0,k,j])
        imNrm=np.linalg.norm(im.ravel(), 2)
        eta = get_eta(im, imNrm, noise_percent, imsz[0])
        cs[n,k,j] = recoverOne(crop(noisy[n,k,j]), imsz, eta, omega, wvlt, solver_mode, omp_mode, omp_iter, omp_ts,omp_usenorms)
        print('Recovered! Repetition: %d, Image: %d, Component: %d' % (n,k,j))

  return_dict[processnum] = cs
  return cs

def recoverAll(fourier_file, orig_file, pattern, noise_percent, c=2, wvlt='haar', 
               mode=CS_MODE, omp_mode=def_omp_mode, omp_iter=def_omp_iter, omp_ts=def_omp_ts, omp_usenorms=def_omp_usenorms):
	
  # Load data
  if isinstance(fourier_file,str):
    data = np.load(fourier_file)
  else:
    data = fourier_file
  
  # Load original image
  if isinstance(orig_file,str):
    original = np.load(orig_file)
  else:
    original = orig_file

  shp = data.shape[0:3] + crop(np.zeros(data.shape[3:])).shape
  recovered = np.empty(shp,dtype=np.complex64)
  interval = max(int(data.shape[0]/c),1)
  manager = Manager()
  return_dict = manager.dict()
  jobs = []
  imsz = crop(original[0,0,0]).shape
  first = original[0,0,0]  
  for n in range(0, data.shape[0], interval):
    if pattern.shape[0] > 1:
      pattern_sample = pattern[n:(n+interval)]
    else:
      pattern_sample = pattern
    p = Process(target=recover, args=(data[n:n+interval], original, pattern_sample, noise_percent, int(n/interval), return_dict, 
                                      wvlt, mode, omp_mode, omp_iter, omp_ts, omp_usenorms))
    jobs.append(p)
    p.start()
  for job in jobs:
    job.join()
  recovered = np.concatenate([v for k,v in sorted(return_dict.items())], axis=0)
  print('Finished recovering, with final shape', recovered.shape)
  return recovered

def recover_vel(compleximg, venc=None, threshold=True):
  vel = np.zeros(compleximg.shape)
  for n in range(0,compleximg.shape[0]):
    for j in range(0, compleximg.shape[2]):
      m = compleximg[n,0,j]
      vel[n,0,j] = np.abs(m)
      for k in range(1,compleximg.shape[1]):
        v = compleximg[n,k,j]
        # Formula in the paper
        v = venc/(np.pi)*(np.angle(v) - np.angle(m))
        vel[n,k,j] = v
  return vel

def phase_info(compleximg):
  #return reference phase and magnitude of velocities
  #required for inversion from velocity components
  refphase = np.angle(compleximg[:,0,:])
  return refphase, np.abs(compleximg)

def linear_reconstruction(fourier_file, omega=None):
  # Check if fourier_file is file or string
  if isinstance(fourier_file, str): 
    kspace = np.load(fourier_file)
  else:
    kspace = fourier_file

  # Check Omega if 2D or 3D with multiple undersampling patterns
  if omega is not None:
    # Check the size of the 3D mask, either one or equal to the samples
    if(len(omega.shape) == 3):
      if((omega.shape[0] != 1)and(omega.shape[0] != kspace.shape[0])):
        print('Invalid number of samples in 3D mask.')
        exit(-1)
    # Multiple mask case
    if(len(omega.shape) == 3):
      for loopA in range(omega.shape[0]):      
        omega[loopA] = crop(omega[loopA])
    elif(len(omega.shape) == 2):
      omega = crop(omega)
    else:
      print("ERROR: mask has unexpected shape.")
      sys.exit(-1)

  # Perform linear reconstructions      
  imsz = crop(kspace[0,0,0]).shape
  kspace = kspace[:,:,:, :imsz[0], :imsz[1]]
  linrec = np.zeros(kspace.shape[0:3] + imsz, dtype=complex)
  for n in range(kspace.shape[0]):
    for k in range(kspace.shape[1]):
      for j in range(kspace.shape[2]):
        if omega is not None:
          if(len(omega.shape) == 2):
            kspace[n,k,j][~omega] = 0
          elif((len(omega.shape) == 3)and(omega.shape[0] == 1)):
            kspace[n,k,j][~omega[0]] = 0
          else:
            kspace[n,k,j][~omega[n]] = 0
        linrec[n,k,j] = fft.ifft2(crop(kspace[n,k,j]), norm='ortho')
  return linrec

def getMethodString(mtd):
  if(mtd == 0):
    return 'CS'
  elif(mtd == 1):
    return 'CSDEB'
  elif(mtd == 2):
    return 'OMP'

# Get File Names
def getFiles(args):
  
  # Noisy Fourier measurement file
  fourier_file = args.fromdir + 'noisy_noise' + str(int(args.noisepercent*100)) + '_n' + str(args.repetitions) + '.npy'

  # Undersampling mask file
  if not args.usemultipatterns: 
    mask_file = args.maskdir + 'undersamplpattern_p' + str(int(args.uVal*100)) + args.uType + '.npy'
  else:
    mask_file = args.maskdir + 'undersamplpattern_p' + str(int(args.uVal*100)) + args.uType + '_n' + str(args.repetitions) + '.npy'
  
  # Image file
  orig_file = args.fromdir + 'imgs_n1.npy'

  # Reconstruction file
  rfs = str(int(args.noisepercent*100)) + '_p' + str(int(args.uVal*100)) + args.uType + '_n' + str(args.repetitions) + '_w' + args.wavelet.upper() + '_a' + getMethodString(args.method) + '.npy' 

  rec_file     = args.recdir + 'rec_noise' + rfs  
  # Save linear reconstruction file and velocities if requested
  rec_lin_file = None
  vel_file     = None
  vel_lin_file = None
  if(args.evallinrec):
    rec_lin_file = args.recdir + 'lin_noise' + rfs    
  if(args.savevels):
    vel_file = args.recdir + 'rec_vel_noise' + rfs  
    if(args.evallinrec):
      vel_lin_file = args.recdir + 'lin_vel_noise' + rfs  
  
  # Return file names
  return fourier_file,mask_file,orig_file,rec_file,vel_file,rec_lin_file,vel_lin_file

# MAIN 
if __name__ == '__main__':

  parser = argparse.ArgumentParser(description='.')

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
    
  # uVal            = 0.75,
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

  # uType           = 'vardengauss',
  parser.add_argument('-u', '--utype',
                      action=None,
                      # nargs='+',
                      const=None,
                      default='vardengauss',
                      type=str,
                      choices=['bernoulli','vardentri','vardengauss','vardenexp','halton'],
                      required=False,
                      help='undersampling pattern type',
                      metavar='',
                      dest='uType')

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

  # numProcesses = 2
  parser.add_argument('-c', '--numprocesses',
                      action=None,
                      # nargs='+',
                      const=None,
                      default=2,
                      type=int,
                      choices=None,
                      required=False,
                      help='number of threads for parallel image reconstruction',
                      metavar='',
                      dest='numprocesses')

  # fromdir         = '\.'
  parser.add_argument('-f', '--fromdir',
                      action=None,
                      # nargs='+',
                      const=None,
                      default='./',
                      type=str,
                      choices=None,
                      required=False,
                      help='folder for original image, k-space images and velocity encoding',
                      metavar='',
                      dest='fromdir')

  # recdir         = '\.'
  parser.add_argument('-d', '--recdir',
                      action=None,
                      # nargs='+',
                      const=None,
                      default='./',
                      type=str,
                      choices=None,
                      required=False,
                      help='folder for recontructed images',
                      metavar='',
                      dest='recdir')

  # maskdir         = '\.'
  parser.add_argument('-m', '--maskdir',
                      action=None,
                      # nargs='+',
                      const=None,
                      default='./',
                      type=str,
                      choices=None,
                      required=False,
                      help='folder for undersampling patterns',
                      metavar='',
                      dest='maskdir')

  # method
  parser.add_argument('-rm', '--method',
                      action=None,
                      # nargs='*',
                      const=None,
                      default=0,
                      type=int,
                      choices=[0,1,2],
                      required=False,
                      help='nonlinear reconstruction method',
                      metavar='',
                      dest='method')

  # Type of OMP solver
  parser.add_argument('--ompmode',
                      action=None,
                      # nargs='*',
                      const=None,
                      default='stomp',
                      type=str,
                      choices=['stomp','omp'],
                      required=False,
                      help='Greedy reconstruction method, either omp or stomp',
                      metavar='',
                      dest='ompmode')

  # omp threshold factor
  parser.add_argument('--ompts',
                      action=None,
                      const=None,
                      default=2.0,
                      type=float,
                      choices=None,
                      required=False,
                      help='threshold factor in stomp',
                      metavar='',
                      dest='ompts')
  
  # Maximum number of iterations for OMP/STOMP
  parser.add_argument('--ompiter',
                      action=None,
                      # nargs='+',
                      const=None,
                      default=10,
                      type=int,
                      choices=None,
                      required=False,
                      help='maximum number of iterations for OMP/STOMP',
                      metavar='',
                      dest='ompiter')

    # save velocities
  parser.add_argument('--ompusenorms',
                    action='store_true',
                    default=False,
                    required=False,
                    help='Include column norms in OMP mathing phase',
                    dest='ompusenorms')    

  # wavelet type
  parser.add_argument('-w', '--wavelet',
                      action=None,
                      # nargs='*',
                      const=None,
                      default='haar',
                      type=str,
                      choices=['haar','db4','db8'],
                      required=False,
                      help='wavelet family',
                      metavar='',
                      dest='wavelet')

  # compute linear reconstruction
  parser.add_argument('--evallinrec',
                      action='store_true',
                      default=False,
                      required=False,
                      help='Evaluate linear reconstructions',
                      dest='evallinrec')    
    
  # use unique undesampling pattern for each reconstruction (false) or change every time
  parser.add_argument('-um', '--usemultipatterns',
                      action='store_true',
                      default=False,
                      required=False,
                      help='generate a unique undersampling pattern for each noise realization',
                      dest='usemultipatterns')

  # save velocities
  parser.add_argument('--savevels',
                    action='store_true',
                    default=False,
                    required=False,
                    help='Save velocity fields',
                    dest='savevels')    

  # Parse Commandline Arguments
  args = parser.parse_args()
  
  # Get File Names
  fourier_file,mask_file,orig_file,rec_file,vel_file,rec_lin_file,vel_lin_file = getFiles(args)
  
  # Perform Reconstruction
  if not os.path.exists(rec_file):

    print('Loading undersampling mask...')    
    if os.path.exists(mask_file):
      umask = np.load(mask_file)
    else:
      print('ERROR: Undersampling mask file not found: ',mask_file)
      sys.exit(-1)

    print('Computing reconstructions...')
    recovered = recoverAll(fourier_file, 
                           orig_file, 
                           umask, 
                           args.noisepercent, 
                           c=args.numprocesses, 
                           wvlt=args.wavelet, 
                           mode=args.method,
                           omp_mode=args.ompmode, 
                           omp_iter=args.ompiter, 
                           omp_ts=args.ompts,
                           omp_usenorms=args.ompusenorms)
    print('Saving reconstruction to file: ',rec_file)
    np.save(rec_file, recovered)
  else:
    print('Retrieving recovered images from numpy file: ',rec_file)
    recovered = np.load(rec_file)

  # Linear Reconstructions
  if(args.evallinrec):
    print('Computing linear reconstrucitons...')
    linrec = linear_reconstruction(fourier_file, umask)
    print('Saving linear reconstrucitons to file: ',rec_lin_file)
    np.save(rec_lin_file, linrec)

  # Save velocities
  if(args.savevels): 

    # Load velocity encoding
    print('Loading velocity encoding...')
    vencfile = args.fromdir + 'venc_n1.npy'
    if os.path.exists(vencfile):
      venc = np.load(vencfile)
    else:
      print('WARNING: file for velocity encoding not found: ',vencfile)
      venc = None  

    print('Computing velocities...')  
    recovered = recover_vel(recovered, venc)
    print('Saving velocities to file: ',vel_file) 
    np.save(vel_file, recovered)
    
    if(args.evallinrec):
      linrec = recover_vel(linrec, venc)
      print('Saving linear reconstructed velocities to file: ',vel_lin_file)
      np.save(vel_lin_file, linrec)
