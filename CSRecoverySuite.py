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
import time

def get_umask_string(samptype):
  if(samptype == 'bernoulli'):
    return 'Bernoulli'
  elif(samptype == 'vardengauss'):
    return 'Gauss'
  else:
    print('ERROR: Invalid mask type')
    sys.exit(-1)

def get_method_string(method):
  if(method == 'cs'):
    return 'CS'
  elif(method == 'csdebias'):
    return 'CS+Deb.'
  elif(method == 'omp'):
    return 'OMP'    
  else:
    print('ERROR: Invalid mask type')
    sys.exit(-1)

def toSlice(idx):
  if idx == None:
    return slice(None)
  else:
    return idx

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

def pywt2array( x, shape ):
  # Essentially a breadth-first traversal
  y = np.zeros((0,1))
  for sublist in x:
    for z in sublist:
      y = np.concatenate((y, np.expand_dims(z.ravel(), axis=1)),axis=0)
  y = y.reshape(shape)
  return y

def array2pywt( x ):
  shape = x.shape
  # Assuming both dims power of 2 or one dim is multiple of other (which is power of 2)
  maxlevel = int(math.log(min(shape),2))
  x = x.ravel()
  coeffs = [None]*(maxlevel+1)
  i = len(x)
  if shape[0] > shape[1]:
    initdim1 = int(shape[0]/shape[1])
    initdim2 = 1
  else:
    initdim1 = 1
    initdim2 = int(shape[1]/shape[0])
  coeffs[0] = np.reshape(x[0:initdim1*initdim2],(initdim1,initdim2))
  for k in reversed(range(1,maxlevel+1)):
    dim1 = int(initdim1*math.pow(2, k-1))
    dim2 = int(initdim2*math.pow(2, k-1))
    coeffs[k] = (np.reshape(x[(i-dim1*dim2*3):(i-dim1*dim2*2)], (dim1,dim2)),
                 np.reshape(x[(i-dim1*dim2*2):(i-dim1*dim2)], (dim1,dim2)),
                 np.reshape(x[(i-dim1*dim2):i], (dim1,dim2)))
    i = i - dim1*dim2*3
  return coeffs

# Added the soft-thresholding function
def sft(x,t):
  # Soft-thresholding
  return x * ( 1 - np.minimum( t / np.maximum( np.abs(x), 1E-32 ), 1 ) )

# Changed the name to project_l1_ball for clarity
def project_l1_ball( x, t ):
  xL1Nrm  = la.norm( x.ravel(), 1 );
  if xL1Nrm > t:
    # If the input is outside the l1-ball we project.
    nx  = x.size;
    # Sort magnitudes in decreasing order
    mx  = np.flip( np.sort( np.absolute( x.ravel() ), kind='mergesort' ) );
    # Obtain unique values (this is needed for some borderline cases)
    umx = np.flip( np.unique( mx ) );
    if( t <= umx[0] - umx[1] ):
      # If the radius of the l1-ball is small and the input is far away
      # and the shrinkage is large.
      vp      = umx[0];
      sroot   = vp - t;
      return sft( x, sroot )
    else:
      # This is the average case
      cmx = np.cumsum( mx );
      smx = cmx - np.array( range(1, nx+1) ) * mx;
      if( smx[nx - 1] < t ):
        # This condition handles some cases when the input is close to
        # the boundary and the shrinkage is small.
        sroot   = (xL1Nrm - t) / nx;
        return sft( x, sroot )
      idp,    = np.where( smx > t );
      idp     = idp[0];
      sroot   = (cmx[idp - 1] - t) / idp;
      return sft( x, sroot )
  else:
    # If the input is within the l1-ball we do nothing.
    return x

class genericOperator(object):
  pass

class OperatorLinear(genericOperator):
  def __init__(self, mat):
    self.__mat = mat
    self.__shape = mat.shape

  @property
  def shape(self):
    "The shape of the operator."
    return self.__shape

  def input_size(self):
    return self.__shape[1]

  @property
  def matrix(self):
    "The shape of the operator."
    return self.__mat

  @property
  def T(self):
    "Transposed of the operator."
    # return OperatorLinear(self.__mat.T)
    return OperatorLinear(np.conjugate(self.__mat.T))

  def __mul_scalar(self,x):
    return self.__mat * x

  def __mul_vector(self,x):
    return np.dot(self.__mat,x)

  def __mul__(self, x):
    "Multiplication"
    if np.isscalar(x):
      return self.__mul_scalar(x)
    if isinstance(x, np.ndarray):
      return self.__mul_vector(x)
    raise ValueError('Cannot multiply')

  def colRestrict(self,idxSet=None):
    return OperatorLinear(self.__mat[:,idxSet])

# Defines a class for linear transforms
class Operator4dFlow(genericOperator):
  
  def __init__(self, insz=None, imsz=None, samplingSet=None, basisSet=None, isTransposed=False,waveletName='haar', waveletMode='periodization'):
    # insz is the size of the array used as input
    self.insz           = insz
    self.imsz           = imsz
    self.waveletName    = waveletName
    self.waveletMode    = waveletMode
    self.isTransposed   = isTransposed
    # SamplingSet is the set of indices that are measured after the mapping
    # has been applied
    self.samplingSet    = samplingSet
    if( samplingSet is None ):
      self.outsz          = imsz
      self._buffer        = None
    else:
      self.outsz          = ( np.count_nonzero( samplingSet ), )
      self._buffer        = np.zeros( self.imsz ) + 1j * np.zeros( self.imsz )
    # Restrict the set of wavelet basis
    self.basisSet = basisSet
    # Constant
    self._cst           = math.pow( np.prod( insz ), -1/2 );

  def eval(self, x, mode):
    '''
    The method eval implements
      obj.eval(x, 1) returns A * x
      obj.eval(x, 2) returns A' * x where A' is the adjoint
    '''
    if( mode == 1 ): # FORWARD MAP
      if( self.samplingSet is None ):
        return self._cst * fft.fft2(pywt.waverec2(array2pywt(x), wavelet=self.waveletName, mode=self.waveletMode));
      else:
        y = self._cst * fft.fft2(pywt.waverec2(array2pywt(x), wavelet=self.waveletName, mode=self.waveletMode));
        return y[ self.samplingSet ];
    if( mode == 2 ): # ADJOINT MAP
      if( self.samplingSet is None ):
        arr = np.conj( fft.fft2( np.conj(x) ) )
        return self._cst * pywt2array(pywt.wavedec2(arr, wavelet=self.waveletName, mode=self.waveletMode), arr.shape);
      else:
        #y = np.zeros( self.imsz ) + 1j * np.zeros( self.imsz );
        self._buffer[ self.samplingSet ] = x[ : ];
        arr = np.conj( fft.fft2( np.conj(self._buffer) ) )
        return self._cst * pywt2array(pywt.wavedec2(arr, wavelet=self.waveletName, mode=self.waveletMode),arr.shape);

  #  The method returns the shape of the input array
  def input_size(self):
    if(self.samplingSet is None):
      return self.insz
    else:
      return self.samplingSet.shape

  @property
  def shape(self):
    "The shape of the operator."
    if(self.samplingSet is None):
      shapeX = np.prod(self.outsz)
    else:
      # CHECK !!! it doesn't seem right
      shapeX = np.prod(self.samplingSet.shape)

    if(self.basisSet is None):
      shapeY = np.prod(self.insz)
    else:
      shapeY = len(self.basisSet)

    if(self.isTransposed == False):
      return shapeX,shapeY
    else:
      return shapeY,shapeX

  def __mul__(self, x):
    "Multiplication"
    # Convert x from 1D to 2D
    if(not(self.isTransposed)):
      
      # FORWARD OPERATOR
      # print("FORWARD OPERATOR MULTIPLICATION")
      
      # Create zero vector
      inV = np.zeros(np.prod(self.insz),dtype=np.complex)
      
      # Restrict the set of basis on the input vector x
      if(not( self.basisSet is None)):
        inV[self.basisSet] = x[:]
      else:
        inV[:] = x[:]

      # Create a 2D Wavelet Representation
      inV = inV.reshape(self.insz)

      # Compute the vector y
      y = self._cst * fft.fft2(pywt.waverec2(array2pywt(inV), wavelet=self.waveletName, mode=self.waveletMode))

      # Select frequencies as per sampling set
      if( self.samplingSet is None ):
        return y
      else:
        return y[ self.samplingSet ]
    
    else:
      
      # ADJOINT OPERATOR
      # print("ADJOINT OPERATOR MULTIPLICATION")

      # Apply Fourier Transform Only for frequencies in the sampling set
      if( self.samplingSet is None ):
        arr = np.conj( fft.fft2( np.conj(x) ) )
      else:
        self._buffer[ self.samplingSet ] = x[ : ];
        arr = np.conj( fft.fft2( np.conj(self._buffer) ) )

      # print('ARR ',arr)

      # Perform wavelet transform
      res = self._cst * pywt2array(pywt.wavedec2(arr, wavelet=self.waveletName, mode=self.waveletMode),arr.shape)

      # Filter wavelet coefficients as per basisSet
      # print(res.dtype)
      if( self.basisSet is None ):
        return res.ravel()
      else:
        return res.ravel()[self.basisSet]

  @property
  def T(self):
    "Transposed of the operator."
    return Operator4dFlow(insz=self.insz, imsz=self.imsz, samplingSet=self.samplingSet, basisSet=self.basisSet, isTransposed=not(self.isTransposed), waveletName=self.waveletName, waveletMode=self.waveletMode)

  # Computes leading singular value of the linear mapping A via power iteration
  def getNorm(self, maxItns=1E3, dsAbsTol=1E-6 ):
    # Initialize variables
    b   = np.random.normal( size = self.input_size() )
    b   = b / la.norm(b)
    s   = 0
    ds  = np.inf
    itn = 0
    # Power iteration loop
    while( ds > dsAbsTol and itn <= maxItns ):
      # Evaluate xk = A'A x
      bk = self.eval(self.eval(b, 1), 2)
      # Evaluate xk' A'Ax to estimate sigma_max^2
      sk = np.sqrt( np.real( np.vdot( bk.flatten(), b.flatten() ) ) )
      ds = np.abs( sk - s )
      # Normalize singular vector
      b  = bk / la.norm(bk)
      s  = sk
    return s

  # Create An operator with a 
  def colRestrict(self,idxSet=None):
    "Restrict operator column set."
    return Operator4dFlow(insz=self.insz, imsz=self.imsz, samplingSet=self.samplingSet, basisSet=idxSet, isTransposed=self.isTransposed, waveletName=self.waveletName, waveletMode=self.waveletMode)

# Computes leading singular value of the linear mapping A via power iteration
def OperatorNorm( A, maxItns=1E3, dsAbsTol=1E-6 ):
  # Initialize variables
  b   = np.random.normal( size = A.input_size() )
  b   = b / la.norm(b)
  s   = 0
  ds  = np.inf
  itn = 0
  # Power iteration loop
  while( ds > dsAbsTol and itn <= maxItns ):
    # Evaluate xk = A'A x
    bk = A.eval(A.eval(b, 1), 2)
    # Evaluate xk' A'Ax to estimate sigma_max^2
    sk = np.sqrt( np.real( np.vdot( bk.flatten(), b.flatten() ) ) )
    ds = np.abs( sk - s )
    # Normalize singular vector
    b  = bk / la.norm(bk)
    s  = sk
  return s

def OperatorTestAdjoint(A, ntrials=100):
  insz    = A.input_size()
  x       = np.random.normal(size=insz) + 1j * np.random.normal(size=insz)
  Ax      = A.eval(x, 1)
  outsz   = Ax.shape
  err     = np.zeros((ntrials,))
  for It in range(0, ntrials):
    x       = np.random.normal(size=insz) + 1j * np.random.normal(size=insz)
    y       = np.random.normal(size=outsz) + 1j * np.random.normal(size=outsz)
    Ax      = A.eval(x, 1)
    Aty     = A.eval(y, 2)
    yAx     = np.sum(np.conjugate(y.ravel()) * Ax.ravel())
    Atyx    = np.sum(np.conjugate(Aty.ravel()) * x.ravel())
    err[It] = np.absolute( yAx - Atyx )
  print('--- ADJOINT TEST')
  print('Inner product')
  print(' Avg. Error:', np.mean(err))
  print(' St. Dev.:', np.std(err))
  print('----------------')

def LSSolver( y, A, w0, maxItns=1E4, dwAbsTol=1E-5, dfwAbsTol=1E-6 ):
    # Find lipschitz constant
    smax        = OperatorNorm( A )
    L           = 1.01 * math.pow(smax, 2)
    # Initialize variables
    w           = w0
    dwNrm       = np.inf
    itn         = 0
    fw          = 0.5 * math.pow(la.norm((y - A.eval(w, 1)).ravel(), 2), 2)
    # Optimization loop
    while( ( dwNrm > dwAbsTol or np.abs(dfw) > dfwAbsTol ) and itn <= maxItns):
        itn         = itn + 1
        wk          = w - (1/L) * A.eval(A.eval(w, 1) - y, 2)
        fwk         = 0.5 * math.pow(la.norm((y - A.eval(wk, 1)).ravel(), 2), 2)
        dfw         = fwk - fw
        dwNrm       = la.norm( (wk - w).ravel(), 2 )
        w           = wk
        fw          = fwk
    return w, fw

def pgdl1( t, x0, L, f, df, maxItns=1E4, dwAbsTol=1E-4, dfwAbsTol=1E-5, disp=0, printEvery=10 ):
    # Project initial iterate on the l1-ball
    w           = project_l1_ball( x0, t );
    fw          = f( w );
    # Initialize variables
    dwNrm       = np.inf;
    dfw         = np.inf;
    itn         = 0;
    s           = 1;
    v           = w;
    # Optimization loop
    while( ( dwNrm > dwAbsTol or np.abs( dfw ) > dfwAbsTol ) and itn < maxItns):
        itn         = itn + 1;
        if( disp == 3 and itn == 1 ):
            print(' [PGDL1] Initial objective: {:+5.3E}'.format(fw) );
        wk          = project_l1_ball( v - (1/L) * df( v ), t );
        sp          = 0.5 * ( 1 + math.pow(1 + 4 * math.pow( s, 2 ), 1/2) );
        vk          = wk + ( ( s - 1 )/sp ) * ( wk - w );
        fwk         = f( wk );
        pwk         = project_l1_ball( wk - (1/L) * df( wk ), t );
        dwNrm       = la.norm( (pwk - wk).ravel(), 2 );
        dfw         = fwk - fw;
        w           = wk;
        v           = vk;
        fw          = fwk;
        s           = sp;
        if( disp == 3 and ( itn == 1 or np.mod( itn, printEvery ) == 0 ) ):
            print(' [PGDL1] Iteration: {:03d} | Objective: {:+5.3E} | Last step: {:+5.3E} | Last decrease: {:+5.3E}'.format(itn, fw, dwNrm, dfw) );
    if( disp > 0 ):
        print(' [PGDL1 Summary] Last iteration: {:03d} | Optimal value: {:+5.3E} | Last step: {:+5.3E} | Last decrease: {:+5.3E}'.format(itn, fw, dwNrm, dfw) );
    return w, fw

class RootPGDL1():
    def __init__(self, eta=0, f=None, df=None, L=None, tmin=None, xmin=None, tmax=None, xmax=None, maxItns=1E4, dwAbsTol=1E-4, dfwAbsTol=1E-5, disp=0):
        self.eta = eta
        self.funObject = f
        self.gradObject = df
        self.lipschitzConstant = L
        self.tmin = tmin
        self.xmin = xmin
        self.tmax = tmax
        self.xmax = xmax
        self.maxItns = maxItns
        self.dwAbsTol = dwAbsTol
        self.dfwAbsTol = dfwAbsTol
        self.disp = disp
        self.lastOptimalPoint = None
        self.lastOptimalValue = None

    def eval(self, t):
        theta = (t - self.tmin) / (self.tmax - self.tmin)
        if( np.absolute( t - self.tmin ) < np.absolute( t - self.tmax ) ):
            xinit = self.xmin
        else:
            xinit = self.xmax
        w, fw = pgdl1(t, x0=xinit, f=self.funObject, df=self.gradObject, L=self.lipschitzConstant, maxItns=self.maxItns, dwAbsTol=self.dwAbsTol, dfwAbsTol=self.dfwAbsTol, disp=0, printEvery=1E32)
        fw = np.sqrt(2 * fw)
        self.lastOptimalPoint = w
        self.lastOptimalValue = fw
        if( np.absolute( t - self.tmin ) < np.absolute( t - self.tmax ) ):
            self.tmin = t
            self.xmin = w
        else:
            self.tmax = t
            self.xmax = w
        if( self.disp > 0 ):
            print('      Trial value t = {:7.5E} with f(t) = {:+7.5E} (theta = {:7.5E})'.format(t, fw - self.eta, theta) );
        return fw - self.eta

def f( y, A, alpha ):
  return 0.5 * math.pow(la.norm((y - A.eval(alpha, 1)).ravel(), 2), 2)

def f_grad( y, A, alpha ):
  return A.eval(A.eval(alpha, 1) - y, 2)

###############################
#### COMPRESSED SENSING #######
###############################

def CSRecovery(eta, y, A, x0, disp=0, printEvery=0):
    # Parameters
    if( disp == 0 ):
        root_disp = 0
    else:
        root_disp = 1
    yNrm        = la.norm( y.ravel(), 2 )
    if( disp > 0 ):
        print(' [CS RECOVERY]' )
        print('      Error level:        {:5.3E}'.format(eta) )
        print('      l2-norm of y:       {:5.3E}'.format(yNrm) )
    if( yNrm <= eta ):
        return 0, np.zeros( x0.shape )
        print('  Summary')
        print('      Optimal value:      {:5.3E}'.format(yNrm) )
        print('      Elapsed time:       {:8.4f} seconds'.format(0.0) )
    # Find lipschitz constant
    smax        = OperatorNorm( A )
    L           = 1.05 * math.pow(smax, 2)
    # Least-squares solution
    wtls, rmax  = LSSolver( y, A, x0 )
    # Lower bound variables
    tmin        = 0
    ftmin       = yNrm
    wtmin       = np.zeros( x0.shape )
    # Find upper bound variables
    wtmax       = wtls
    tmax        = la.norm( wtmax.ravel(), 1 )
    ftmax       = 0
    # Initialize variables
    dt          = tmax - tmin
    dft         = np.abs( ftmin - ftmax )
    if( disp > 0 ):
        print('      Initial tmin:       {:5.3E}'.format(tmin) )
        print('      Initial tmax:       {:5.3E}'.format(tmax) )
    # Root finding via TOMS748
    tend        = time.time()
    froot = RootPGDL1(eta=eta, f=lambda x: f(y, A, x), df=lambda x: f_grad(y, A, x), L=L, tmin=tmin, xmin=wtmin, tmax=tmax, xmax=wtmax, maxItns=1E4, dwAbsTol=1E-4, dfwAbsTol=1E-5, disp=root_disp)
    topt, rnfo = scipy.optimize.toms748(lambda t: froot.eval(t), tmin, tmax, xtol=1E-3, full_output=True, disp=True)
    wopt = froot.lastOptimalPoint
    tend        = time.time() - tend
    if( disp > 0 ):
        print('  Summary');
        print('      Iterations:          {:d}'.format(rnfo.iterations) )
        print('      Function calls:      {:d}'.format(rnfo.function_calls) )
        print('      Optimal value:       {:5.3E}'.format(topt) )
        print('      Elapsed time:        {:8.4f} seconds'.format(tend) )
    return wopt, la.norm( wopt.ravel(), 1 )

def CSRecoveryDebiasing(y, A, x, maxItns=1E4, dwAbsTol=1E-5, dfwAbsTol=1E-6, xthr=1E-6):
    # Find lipschitz constant
    smax        = OperatorNorm( A )
    L           = 1.01 * math.pow(smax, 2)
    # Find support of the input signal
    xsupp       = np.where( np.absolute(x) > xthr, 1, 0)
    # Initialize variables
    w           = x
    dwNrm       = np.inf
    itn         = 0
    fw          = 0.5 * math.pow(la.norm((y - A.eval(w, 1)).ravel(), 2), 2)
    # Optimization loop
    while( ( dwNrm > dwAbsTol or np.abs(dfw) > dfwAbsTol ) and itn <= maxItns):
        itn         = itn + 1
        wk          = w - (1/L) * A.eval(A.eval(w, 1) - y, 2)
        wk          = np.where(xsupp > 0, wk, 0)
        fwk         = 0.5 * math.pow(la.norm((y - A.eval(wk, 1)).ravel(), 2), 2)
        dfw         = fwk - fw
        dwNrm       = la.norm( (wk - w).ravel(), 2 )
        w           = wk
        fw          = fwk
    return w, fw

############################
## SAMPLING FUNCTIONS ######
###########################

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

def generateSamplingMask(imsz, p, saType='bernoulli', num_patterns=1):
  # p is the undersampling ratio: what you don't sample
  if(p < 0.0)or(p > 1.0):
    print('ERROR: Invalid undersampling ratio delta in generateSamplingMask.')
    sys.exit(-1)
  elif(p == 0.0):
    return np.full(imsz, True, dtype=bool)
  else:
    mask = np.empty((num_patterns, ) + imsz, dtype=np.bool)
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

################################
#### OMP RECOVERY   ############
################################

"""
Solve the least-squares problem
  minimize ||Ax-b||
using LSQR.  This is a line-by-line translation from Matlab code
available at http://www.stanford.edu/~saunders/lsqr with several
Pythonic enhancements.
Michael P. Friedlander, University of British Columbia
Dominique Orban, Ecole Polytechnique de Montreal
"""
import sys
import numpy as np
from math import sqrt

__docformat__ = 'restructuredtext'

# Simple shortcuts---linalg.norm is too slow for small vectors
def normof2(x,y): 
  return sqrt(x*x + y*y)

def normof4(x1,x2,x3,x4): 
  return sqrt(x1*x1 + x2*x2 + x3*x3 + x4*x4)

class lsQR(object):
    r"""
    LSQR solves  `Ax = b`  or  `minimize |b - Ax|` in Euclidian norm  if
    `damp = 0`, or `minimize |b - Ax| + damp * |x|` in Euclidian norm if
    `damp > 0`.
    `A`  is an (m x n) linear operator defined by  `y = A * x` (or `y = A(x)`),
    where `y` is the result of applying the linear operator to `x`. Application
    of transpose linear operator must be accessible via `u = A.T * x` (or
    `u = A.T(x)`). The shape of the linear operator `A` must be accessible via
    `A.shape`. A convenient way to achieve this is to make sure that `A` is
    a `LinearOperator` instance.
    LSQR uses an iterative (conjugate-gradient-like) method.
    For further information, see
    1. C. C. Paige and M. A. Saunders (1982a).
       LSQR: An algorithm for sparse linear equations and sparse least
       squares, ACM TOMS 8(1), 43-71.
    2. C. C. Paige and M. A. Saunders (1982b).
       Algorithm 583. LSQR: Sparse linear equations and least squares
       problems, ACM TOMS 8(2), 195-209.
    3. M. A. Saunders (1995).  Solution of sparse rectangular systems using
       LSQR and CRAIG, BIT 35, 588-604.
    integer(ip),  intent(in)  :: m, n, itnlim, nout
    integer(ip),  intent(out) :: istop, itn
    logical,  intent(in)  :: wantse
    complex(dp), intent(in)  :: b(m)
    complex(dp), intent(out) :: x(n)
    real(dp), intent(out) :: se(*)
    real(dp), intent(in)  :: atol, btol, conlim, damp
    real(dp), intent(out) :: Anorm, Acond, rnorm, Arnorm, xnorm       
    """

    def __init__(self, A, **kwargs):

        # Initialize.
        self.name = 'Least-Squares QR'
        self.acronym = 'LSQR'
        self.prefix = self.acronym + ': '

        self.msg=['The exact solution is  x = 0                              ',
                  'Ax - b is small enough, given atol, btol                  ',
                  'The least-squares solution is good enough, given atol     ',
                  'The estimate of cond(Abar) has exceeded conlim            ',
                  'Ax - b is small enough for this machine                   ',
                  'The least-squares solution is good enough for this machine',
                  'Cond(Abar) seems to be too large for this machine         ',
                  'The iteration limit has been reached                      ',
                  'The truncated direct error is small enough, given etol    ']

        self.A = A
        self.x = None ; self.var = None

        self.itn = 0; self.istop = 0
        self.Anorm = 0.; self.Acond = 0. ; self.Arnorm = 0.
        self.xnorm = 0.;
        self.r1norm = 0.; self.r2norm = 0.
        self.optimal = False
        self.resids = []             # Least-squares objective function values.
        self.normal_eqns_resids = [] # Residuals of normal equations.
        self.dir_errors_window = []  # Direct error estimates.
        self.error_upper_bound = []  # Upper bound on direct error.
        self.iterates = []
        return

    def solve(self, b, itnlim=0, damp=0.0, M=None, N=None, atol=1.0e-9,
              btol=1.0e-9, conlim=1.0e+8, show=False, wantvar=False, **kwargs):
        """
        Solve the linear system, linear least-squares problem or regularized
        linear least-squares problem with specified parameters. All return
        values below are stored in members of the same name.
        :parameters:
           :b:    right-hand side vector.
           :itnlim: is an explicit limit on iterations (for safety).
           :damp:   damping/regularization parameter.
        :keywords:
           :atol:
           :btol:  are stopping tolerances.  If both are 1.0e-9 (say),
                   the final residual norm should be accurate to about 9 digits.
                   (The final x will usually have fewer correct digits,
                   depending on `cond(A)` and the size of `damp`.)
           :etol:  stopping tolerance based on direct error (default 1.0e-6).
           :conlim: is also a stopping tolerance.  lsqr terminates if an
                    estimate of `cond(A)` exceeds `conlim`.  For compatible
                    systems `Ax = b`, `conlim` could be as large as 1.0e+12
                    (say).  For least-squares problems, `conlim` should be less
                    than 1.0e+8. Maximum precision can be obtained by setting
                    `atol` = `btol` = `conlim` = zero, but the number of
                    iterations may then be excessive.
           :show:   if set to `True`, gives an iteration log.
                    If set to `False`, suppresses output.
           :store_resids: Store full residual norm history (default: False).
           :window: Number of consecutive iterations over which the director error
                    should be measured (default: 5).
        :return:
           :x:     is the final solution.
           :istop: gives the reason for termination.
           :istop: = 1 means x is an approximate solution to Ax = b.
                   = 2 means x approximately solves the least-squares problem.
           :r1norm: = norm(r), where r = b - Ax.
           :r2norm: = sqrt(norm(r)^2  +  damp^2 * norm(x)^2)
                    = r1norm if damp = 0.
           :Anorm: = estimate of Frobenius norm of (regularized) A.
           :Acond: = estimate of cond(Abar).
           :Arnorm: = estimate of norm(A'r - damp^2 x).
           :xnorm: = norm(x).
           :var:   (if present) estimates all diagonals of (A'A)^{-1}
                   (if damp=0) or more generally (A'A + damp^2*I)^{-1}.
                   This is well defined if A has full column rank or damp > 0.
                   (Not sure what var means if rank(A) < n and damp = 0.)
        """

        etol           = kwargs.get('etol', 1.0e-6)
        store_resids   = kwargs.get('store_resids', False)
        store_iterates = kwargs.get('store_iterates', False)
        window         = kwargs.get('window', 5)

        self.resids             = [] # Least-squares objective function values.
        self.normal_eqns_resids = [] # Residuals of normal equations.
        self.dir_errors_window  = [] # Direct error estimates.
        self.iterates           = []

        A = self.A
        m, n = A.shape

        if itnlim == 0: itnlim = 3*n

        if wantvar:
            var = np.zeros(n,1)
        else:
            var = None

        dampsq = damp*damp

        itn = istop = 0
        ctol = 0.0
        if conlim > 0.0: self.ctol = 1.0/conlim
        Anorm = Acond = 0.
        z = xnorm = xxnorm = ddnorm = res2 = 0.
        cs2 = -1. ; sn2 = 0.

        if show:
            print(' ')
            print('LSQR            Least-squares solution of  Ax = b')
            str1='The matrix A has %8d rows and %8d cols' % (m, n)
            str2='damp = %20.14e     wantvar = %-5s' % (damp, repr(wantvar))
            str3='atol = %8.2e                 conlim = %8.2e' % (atol, conlim)
            str4='btol = %8.2e                 itnlim = %8g' % (btol, itnlim)
            print(str1); print(str2); print(str3); print(str4);

        # Set up the first vectors u and v for the bidiagonalization.
        # These satisfy  beta*M*u = b,  alpha*N*v = A'u.

        x = np.zeros(n,dtype=np.complex)
        xNrgNorm2 = 0.0                          # Squared energy norm of final solution.
        dErr = np.zeros(window)                  # Truncated direct error terms.
        trncDirErr = 0                           # Truncated direct error.

        if store_iterates:
            self.iterates.append(x.copy())

        Mu = b[:m].copy()
        if M is not None:
            u = M(Mu)
        else:
            u = Mu

        alpha = 0.
        # beta = np.sqrt(np.dot(u,Mu))       # norm(u)
        beta = np.linalg.norm(u)
        if beta > 0:
            u /= beta
            if M is not None: Mu /= beta

            Nv = A.T * u
            if N is not None:
                v = N(Nv)
            else:
                v = Nv
            # alpha = np.sqrt(np.dot(v,Nv))   # norm(v)
            alpha = np.linalg.norm(v)

        if alpha > 0:
            v /= alpha
            if N is not None: Nv /= alpha
            w = v.copy()     # Should this be Nv ???

        x_is_zero = False   # Is x=0 the solution to the least-squares prob?

        Arnorm = alpha * beta
        if Arnorm == 0.0:
            if show: print(self.msg[0])
            x_is_zero = True
            istop = 0

        rhobar = alpha
        phibar = beta
        bnorm  = beta
        rnorm  = beta

        r1norm = rnorm
        r2norm = rnorm
        head1  = '   Itn      x(1)       r1norm     r2norm '
        head2  = ' Compatible   LS      Norm A   Cond A'

        if show:
            print(' ')
            print(head1+head2)
            test1  = 1.0
            test2  = alpha / beta if not x_is_zero else 1.0
            str1   = '%6g %12.5e'     % (itn,    np.abs(x[0]))
            str2   = ' %10.3e %10.3e' % (r1norm, r2norm)
            str3   = '  %8.1e %8.1e'  % (test1,  test2)
            print(str1+str2+str3)

        if store_resids:
            self.resids.append(r2norm)
            self.normal_eqns_resids.append(Arnorm)

        # ------------------------------------------------------------------
        #     Main iteration loop.
        # ------------------------------------------------------------------
        while itn < itnlim and not x_is_zero:

            itn = itn + 1

            #   Perform the next step of the bidiagonalization to obtain the
            #   next  beta, u, alpha, v.  These satisfy the relations
            #               beta*M*u  =  A*v   -  alpha*M*u,
            #              alpha*N*v  =  A'*u  -   beta*N*v.

            Mu = A*v - alpha*Mu
            if M is not None:
                u = M(Mu)
            else:
                u = Mu
            # beta = np.sqrt(np.dot(u,Mu))   # norm(u)
            beta = np.linalg.norm(u)

            if beta > 0:
                u /= beta
                if M is not None: Mu /= beta
                Anorm = normof4(Anorm, alpha, beta, damp)
                Nv = A.T*u - beta*Nv
                if N is not None:
                    v = N(Nv)
                else:
                    v = Nv
                # alpha = np.sqrt(np.dot(v,Nv))  # norm(v)
                alpha = np.linalg.norm(v)
                if alpha > 0:
                    v /= alpha
                    if N is not None: Nv /= alpha

            # Use a plane rotation to eliminate the damping parameter.
            # This alters the diagonal (rhobar) of the lower-bidiagonal matrix.

            rhobar1 = normof2(rhobar, damp)
            cs1     = rhobar / rhobar1
            sn1     = damp   / rhobar1
            psi     = sn1 * phibar
            phibar  = cs1 * phibar

            #  Use a plane rotation to eliminate the subdiagonal element (beta)
            # of the lower-bidiagonal matrix, giving an upper-bidiagonal matrix.

            rho     =   normof2(rhobar1, beta)
            cs      =   rhobar1 / rho
            sn      =   beta    / rho
            theta   =   sn * alpha
            rhobar  = - cs * alpha
            phi     =   cs * phibar
            phibar  =   sn * phibar
            tau     =   sn * phi

            # Update x and w.

            t1      =   phi   / rho
            t2      = - theta / rho
            dk      =   (1.0/rho)*w
            x      += t1*w
            w      *= t2
            w      += v

            ddnorm  = ddnorm + np.linalg.norm(dk)**2
            if wantvar: var += dk*dk

            if store_iterates:
                self.iterates.append(x.copy())

            # Update energy norm of x.
            xNrgNorm2 += phi*phi
            dErr[itn % window] = phi
            if itn > window:
                trncDirErr = np.linalg.norm(dErr)
                xNrgNorm = sqrt(xNrgNorm2)
                self.dir_errors_window.append(trncDirErr / xNrgNorm)
                if trncDirErr < etol * xNrgNorm:
                    istop = 8

            # Use a plane rotation on the right to eliminate the
            # super-diagonal element (theta) of the upper-bidiagonal matrix.
            # Then use the result to estimate norm(x).

            delta   =   sn2 * rho
            gambar  = - cs2 * rho
            rhs     =   phi  -  delta * z
            # print(phi)
            # sys.exit(-1)
            zbar    =   rhs / gambar
            xnorm   =   sqrt(xxnorm + zbar**2)
            gamma   =   normof2(gambar, theta)
            cs2     =   gambar / gamma
            sn2     =   theta  / gamma
            z       =   rhs    / gamma
            # print(rhs)
            # sys.exit(-1)
            xxnorm +=   z*z

            # Test for convergence.
            # First, estimate the condition of the matrix  Abar,
            # and the norms of  rbar  and  Abar'rbar.

            Acond   =   Anorm * sqrt(ddnorm)
            res1    =   phibar**2
            res2    =   res2  +  psi**2
            rnorm   =   sqrt(res1 + res2)
            Arnorm  =   alpha * abs(tau)

            # 07 Aug 2002:
            # Distinguish between
            #    r1norm = ||b - Ax|| and
            #    r2norm = rnorm in current code
            #           = sqrt(r1norm^2 + damp^2*||x||^2).
            #    Estimate r1norm from
            #    r1norm = sqrt(r2norm^2 - damp^2*||x||^2).
            # Although there is cancellation, it might be accurate enough.

            r1sq    =   rnorm**2  -  dampsq * xxnorm
            r1norm  =   sqrt(abs(r1sq))
            if r1sq < 0: r1norm = - r1norm
            r2norm  =   rnorm

            # print('res1 ',r1norm)
            # print('resNorm ',np.linalg.norm(b-A*x))
            # sys.exit(-1)

            # Now use these norms to estimate certain other quantities,
            # some of which will be small near a solution.

            test1 = rnorm / bnorm
            if Anorm == 0. or rnorm == 0.:
                test2 = inf
            else:
                test2 = Arnorm/(Anorm * rnorm)

            if Acond == 0.0:
                test3 = inf
            else:
                test3 = 1.0 / Acond
            t1    = test1 / (1    +  Anorm * xnorm / bnorm)
            rtol  = btol  +  atol *  Anorm * xnorm / bnorm

            if store_resids:
                self.resids.append(r2norm)
                self.normal_eqns_resids.append(Arnorm)

            # The following tests guard against extremely small values of
            # atol, btol  or  ctol.  (The user may have set any or all of
            # the parameters  atol, btol, conlim  to 0.)
            # The effect is equivalent to the normal tests using
            # atol = eps,  btol = eps,  conlim = 1/eps.

            if itn >= itnlim:  istop = 7
            if 1 + test3 <= 1: istop = 6
            if 1 + test2 <= 1: istop = 5
            if 1 + t1    <= 1: istop = 4

            # Allow for tolerances set by the user.

            if test3 <= ctol: istop = 3
            if test2 <= atol: istop = 2
            if test1 <= rtol: istop = 1

            # See if it is time to print something.

            prnt = False;
            if n     <= 40       : prnt = True
            if itn   <= 10       : prnt = True
            if itn   >= itnlim-10: prnt = True
            if itn % 10 == 0     : prnt = True
            if test3 <=  2*ctol  : prnt = True
            if test2 <= 10*atol  : prnt = True
            if test1 <= 10*rtol  : prnt = True
            if istop !=  0       : prnt = True

            if prnt and show:
                str1 = '%6g %12.5e'     % (  itn,   np.abs(x[0]))
                str2 = ' %10.3e %10.3e' % (r1norm, r2norm)
                str3 = '  %8.1e %8.1e'  % (test1,  test2)
                str4 = ' %8.1e %8.1e'   % (Anorm,  Acond)
                print(str1+str2+str3+str4)

            if istop > 0: break

            # End of iteration loop.
            # Print the stopping condition.

        if show:
            print(' ')
            print('LSQR finished')
            print(self.msg[istop])
            print(' ')
            str1 = 'istop =%8g   r1norm =%8.1e'   % (istop, r1norm)
            str2 = 'Anorm =%8.1e   Arnorm =%8.1e' % (Anorm, Arnorm)
            str3 = 'itn   =%8g   r2norm =%8.1e'   % (itn,   r2norm)
            str4 = 'Acond =%8.1e   xnorm  =%8.1e' % (Acond, xnorm)
            str5 = '                  bnorm  =%8.1e'    % bnorm
            str6 = 'xNrgNorm2 = %7.1e   trnDirErr = %7.1e' % \
                    (xNrgNorm2, trncDirErr)
            print(str1 + '   ' + str2)
            print(str3 + '   ' + str4)
            print(str5)
            print(str6)
            print(' ')

        if istop == 0: self.status = 'solution is zero'
        if istop in [1,2,4,5]: self.status = 'residual small'
        if istop in [3,6]: self.status = 'ill-conditioned operator'
        if istop == 7: self.status = 'max iterations'
        if istop == 8: self.status = 'direct error small'
        self.optimal = istop in [1,2,4,5,8]
        self.x = self.bestSolution = x
        self.istop = istop
        self.itn = itn
        self.nMatvec = 2*itn
        self.r1norm = r1norm
        self.r2norm = r2norm
        self.residNorm = r2norm
        self.Anorm = Anorm
        self.Acond = Acond
        self.Arnorm = Arnorm
        self.xnorm = xnorm
        self.var = var
        return

def OMPRecovery(A, b, tol=1E-6, fastAlg=True, showProgress=True, progressInt=10, maxItns=None):

  #gives the OMP solution x given the matrix A and vector b and error 
  #parameter, Tr, to find a stopping point
  m,n = A.shape
  if maxItns is None:
    maxItns=m

  # Create a new vector for the residual starting from b
  curr_res = b.copy()
  iniResNorm = np.linalg.norm(curr_res, ord=2)

  # Intialize Empty index set
  indexSet = []
  notIndexSet = [loopA for loopA in range(n)]

  # Initialize Solution to Zero
  ompSol = np.zeros(n,dtype=np.complex) 
  
  # Determine Column Norms
  if(not(fastAlg)):
    Anorm = np.zeros(n,dtype=np.complex)
    # print('Computing Operator Column Norms...',end=' '); sys.stdout.flush()
    for loopA in range(n):
      unit = np.zeros(n,dtype=np.complex)
      unit[loopA] = 1.0 + 0.0j
      Anorm[loopA] = np.linalg.norm(A * unit)**2
      # print('%15d %15f' % (loopA,Anorm[loopA]))
    Anorm[0] = 1.0 + 0.0j
    # print('OK'); sys.stdout.flush()

  # Print Header
  if(showProgress):
    print('%10s %15s' % ('OMP Iter.','Res. Norm'))

  # OMP Main Loop 
  Finished = False
  count = 0 # Init Counter
  while (not(Finished)):

    # Init counters
    count += 1
    selectedCol = 0
    
    # Start with the Norm of b
    minepsilon = np.linalg.norm(b, ord=2)**2

    if(not(fastAlg)):
      # print('Computing vector e...',end=' '); sys.stdout.flush()
      # Compute vector z
      z = (A.T * curr_res)/Anorm
      # Compute the vector e
      e = np.zeros(n)
      for loopA in range(n):
        unit = np.zeros(n,dtype=np.complex)
        unit[loopA] = z[loopA]
        e[loopA] = np.linalg.norm((A * unit) - curr_res)**2
      # print('OK'); sys.stdout.flush()
      # Get the minimum of e restricted to the indices not in the support
      e[indexSet] = np.inf
      selectedCol = np.argmin(e)
    else:
      selectedCol = notIndexSet[np.argmax(np.absolute(A.T * curr_res)[notIndexSet])]
              
    # Add index to index set
    indexSet.append(selectedCol)

    # Remove from not index set
    notIndexSet.remove(selectedCol)

    # Initialize lsQR
    B = A.colRestrict(indexSet)
    lsqr = lsQR(B)
    
    # Solve least Squares    
    # lsqr.solve(b, show=True)
    # print('LSQR...',end=' '); sys.stdout.flush()
    lsqr.solve(b)
    # print('OK'); sys.stdout.flush()
    ompSol[indexSet] = lsqr.x

    # Update residual    
    curr_res = b - A * ompSol

    # Compute relative residual norm
    resNorm = np.linalg.norm(curr_res, ord = 2)/iniResNorm

    # Print Message
    if(showProgress):
      if(count % progressInt == 0)and(count >= progressInt):
        print('%10d %15e' % (count,resNorm))
    
    # Check finished
    if (resNorm < tol or count > maxItns):
      Finished = True
    
  # Return Result
  return ompSol.reshape(A.input_size()), la.norm(ompSol.ravel(),1)
