import numpy.fft as fft
import pywt
import numpy as np
import sys
import numpy.linalg as la
from scipy import optimize
from scipy.optimize import toms748
from scipy.stats import bernoulli
import scipy
import math

#######################
## VARIOUS UTILITIES ##
#######################

def get_umask_string(samptype):
  if(samptype == 'bernoulli'):
    return 'Bernoulli'
  elif(samptype == 'vardengauss'):
    return 'Gauss'
  else:
    print('ERROR: Invalid mask type')
    sys.exit(-1)

def extractFluidMask(img):  
  res = np.max(np.absolute(img),axis=0)[0,0]
  return (res > 0)    

def get_method_string(method):
  if(method == 'cs'):
    return 'CS'
  elif(method == 'csdebias'):
    return 'CSDEB'
  elif(method == 'omp'):
    return 'OMP'    
  else:
    print('ERROR: Invalid recovery method')
    sys.exit(-1)

def get_wavelet_string(wavelet):
  if(wavelet == 'haar'):
    return 'HAAR'
  elif(wavelet == 'db8'):
    return 'DB8'
  else:
    print('ERROR: Invalid wavelet type')
    sys.exit(-1)

# Methods for cropping in the case the dimensions aren't a power of 2
def powerof2(num):
  # Return the highest power of 2 less than or equal to number
  return int(math.pow(2,math.floor(math.log(num, 2))))

def crop(x):
  if x.shape[1] > x.shape[0]:
    dim1 = powerof2(x.shape[0])
    #max between power of 2 or multiple of other dimension
    dim2 = max(powerof2(x.shape[1]), math.floor(x.shape[1]/dim1)*dim1)
  else:
    dim2 = powerof2(x.shape[1])
    dim1 = max(powerof2(x.shape[0]), math.floor(x.shape[0]/dim2)*dim2)
  return x[0:dim1, 0:dim2]

def isvalidcrop(x):
  '''
  Check if the image has valid power of 2 dimensions
  '''
  y = x.copy()
  y = crop(y)
  if(x.shape[0] == y.shape[0])and(x.shape[1] == y.shape[1]):
    return True
  else:
    return False

########################
## SAMPLING FUNCTIONS ##
########################

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

def VardensSampling(shape, f):
    x = np.linspace(-1, 1, num=shape[1])
    y = np.linspace(-1, 1, num=shape[0])
    omega = np.full(shape, False)
    for Ix in range(0, shape[1]):
        for Iy in range(0, shape[0]):
            u = np.random.uniform(0, 1)
            if( u < f([ x[Ix], y[Iy] ]) ):
                omega[Ix, Iy] = True
    return np.fft.fftshift(omega)

def VardensTriangleSampling(shape, delta):
    if( delta < 0.25 ):
        c = delta / 0.25
        a = 0
    else:
        c = 1
        a = 2 * np.sqrt(delta) - 1
    x = np.linspace(-1, 1, num=shape[1])
    y = np.linspace(-1, 1, num=shape[0])
    omega = np.full(shape, False)
    for Ix in range(0, shape[1]):
        for Iy in range(0, shape[0]):
            if( np.max([ np.abs(x[Ix]), np.abs(y[Iy]) ]) < a ):
                omega[Ix, Iy] = True
            else:
                p = c
                if( np.abs(x[Ix]) > a ):
                    p = p * (1 - (np.abs(x[Ix]) - a) / (1 - a))
                if( np.abs(y[Iy]) > a ):
                    p = p * (1 - (np.abs(y[Iy]) - a) / (1 - a))
                u = np.random.uniform(0, 1)
                if( u < p ):
                    omega[Ix, Iy] = True
    return np.fft.fftshift(omega)

def VardensGaussianSampling(shape, delta):
    c = 2 * np.sqrt(delta / np.pi)
    s, rnfo = scipy.optimize.toms748(lambda t: scipy.special.erf(t) - c * t, 1E-6, 1/c, xtol=1E-3, full_output=True, disp=True)
    s = 1 / (s * np.sqrt(2))
    x = np.linspace(-1, 1, num=shape[1])
    y = np.linspace(-1, 1, num=shape[0])
    omega = np.full(shape, False)
    for Ix in range(0, shape[1]):
        for Iy in range(0, shape[0]):
            p = np.exp( -(x[Ix]**2 + y[Iy]**2) / (2 * s**2) )
            u = np.random.uniform(0, 1)
            if( u < p ):
                omega[Ix, Iy] = True
    return np.fft.fftshift(omega)

def VardensExponentialSampling(shape, delta):
    c = np.sqrt(delta)
    a, rnfo = scipy.optimize.toms748(lambda t: np.exp(-t) - 1 + c * t, 1E-6, 1/c, xtol=1E-3, full_output=True, disp=True)
    x = np.linspace(-1, 1, num=shape[1])
    y = np.linspace(-1, 1, num=shape[0])
    omega = np.full(shape, False)
    for Ix in range(0, shape[1]):
        for Iy in range(0, shape[0]):
            p = np.exp( -a * (np.abs(x[Ix]) + np.abs(y[Iy])) )
            u = np.random.uniform(0, 1)
            if( u < p ):
                omega[Ix, Iy] = True
    return np.fft.fftshift(omega)   

def bisect(f, a, b, tol=1E-3):
  #bisect (binary search) for function only defined for integers
  while f(a) < f(b):
    mid = int((a+b)/2)
    midval = f(mid)
    if midval < 0-tol:
      a = mid + 1
    elif midval > 0+tol:
      b = mid
    else: #midval within (-tol, tol)
      return mid
  #root not found, but return closest pt anyway 
  if np.abs(f(a)) < np.abs(f(b)):
    return a
  return b

def HaltonSampling(shape, p):
  def getPoints(numPts):
    pts = np.transpose(np.asarray(halton_sequence(numPts, 2)))
    pts[:,0] = pts[:,0]*shape[0]
    pts[:,1] = pts[:,1]*shape[1]
    pts = pts.astype(int)
    return pts
  def getRatio(numPts):
    pts = getPoints(numPts)
    return 1.0*np.unique(pts, axis=0).shape[0]/(np.prod(shape))
  f = lambda x: getRatio(x) - p
  numPts = bisect(f, int(p*np.prod(shape)), 4*np.prod(shape)) #f: increasing fcn
  pts = getPoints(numPts)
  indices = np.zeros(shape)
  indices[pts[:,0], pts[:,1]] = 1
  return indices

def generateSamplingMask(imsz, p, saType='bernoulli', num_patterns=1, seed=1234321):
  # p is the undersampling ratio: what you don't sample
  if(p < 0.0)or(p > 1.0):
    print('ERROR: Invalid undersampling ratio delta in generateSamplingMask.')
    sys.exit(-1)
  elif(p == 0.0):
    return np.full(imsz, True, dtype=bool)
  else:
    np.random.seed(seed)
    mask = np.empty((num_patterns, ) + imsz, dtype=bool)
    for k in range(num_patterns):
      #to keep undersampling the same for each slice
      if saType=='bernoulli':
        indices = bernoulli.rvs(size=(imsz), p=(1-p))
      elif saType =='vardentri':
        indices = VardensTriangleSampling(imsz, (1-p))
      elif saType =='vardengauss': #gaussian density
        indices = VardensGaussianSampling(imsz, (1-p))
      elif saType == 'vardenexp': #exponential density
        indices = VardensExponentialSampling(imsz, (1-p))
      elif saType == 'halton': #halton sequence
        indices = HaltonSampling(imsz, (1-p))
      else:
        print('ERROR: Invalid sampling type')
        sys.exit(-1)
      mask[k] = np.ma.make_mask(indices)
      # Return the complement of the mask
    return mask
