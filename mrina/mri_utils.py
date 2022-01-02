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

# Defines a class for linear transforms
class Operator4dFlow(genericOperator):
  
  def __init__(self, insz=None, imsz=None, samplingSet=None, basisSet=None, isTransposed=False, waveletName='haar', waveletMode='periodization'):
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
    # self._cst           = math.pow( np.prod( insz ), -1/2 );

  def eval(self, x, mode):
    '''
    The method eval implements
      obj.eval(x, 1) returns A * x
      obj.eval(x, 2) returns A' * x where A' is the adjoint
    '''
    if( mode == 1 ): # FORWARD MAP
      if( self.samplingSet is None ):
        # return self._cst * fft.fft2(pywt.waverec2(array2pywt(x), wavelet=self.waveletName, mode=self.waveletMode));
        return fft.fft2(pywt.waverec2(array2pywt(x), wavelet=self.waveletName, mode=self.waveletMode),norm='ortho');
      else:
        # y = self._cst * fft.fft2(pywt.waverec2(array2pywt(x), wavelet=self.waveletName, mode=self.waveletMode));
        y = fft.fft2(pywt.waverec2(array2pywt(x), wavelet=self.waveletName, mode=self.waveletMode),norm='ortho');
        return y[ self.samplingSet ];
    if( mode == 2 ): # ADJOINT MAP
      if( self.samplingSet is None ):
        # arr = np.conj( fft.fft2( np.conj(x) ) )
        # return self._cst * pywt2array(pywt.wavedec2(arr, wavelet=self.waveletName, mode=self.waveletMode), arr.shape);
        arr = np.conj( fft.fft2( np.conj(x),norm='ortho' ) )
        return pywt2array(pywt.wavedec2(arr, wavelet=self.waveletName, mode=self.waveletMode), arr.shape);
      else:
        #y = np.zeros( self.imsz ) + 1j * np.zeros( self.imsz );
        self._buffer[ self.samplingSet ] = x[ : ];
        # arr = np.conj( fft.fft2( np.conj(self._buffer) ) )
        # return self._cst * pywt2array(pywt.wavedec2(arr, wavelet=self.waveletName, mode=self.waveletMode),arr.shape);
        arr = np.conj( fft.fft2( np.conj(self._buffer),norm='ortho' ) )
        return pywt2array(pywt.wavedec2(arr, wavelet=self.waveletName, mode=self.waveletMode),arr.shape);


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
      # y = self._cst * fft.fft2(pywt.waverec2(array2pywt(inV), wavelet=self.waveletName, mode=self.waveletMode))
      y = fft.fft2(pywt.waverec2(array2pywt(inV), wavelet=self.waveletName, mode=self.waveletMode), norm='ortho')

      # Select frequencies as per sampling set
      if( self.samplingSet is None ):
        return y
      else:
        return y[ self.samplingSet ]
    
    else:
    
      # Apply Fourier Transform Only for frequencies in the sampling set
      if( self.samplingSet is None ):
        # arr = np.conj( fft.fft2( np.conj(x) ) )
        arr = np.conj( fft.fft2( np.conj(x), norm='ortho' ) )
      else:
        self._buffer[ self.samplingSet ] = x[ : ];
        # arr = np.conj( fft.fft2( np.conj(self._buffer) ) )
        arr = np.conj( fft.fft2( np.conj(self._buffer), norm='ortho' ) )

      # Perform wavelet transform
      # res = self._cst * pywt2array(pywt.wavedec2(arr, wavelet=self.waveletName, mode=self.waveletMode),arr.shape)
      res = pywt2array(pywt.wavedec2(arr, wavelet=self.waveletName, mode=self.waveletMode),arr.shape)

      # Filter wavelet coefficients as per basisSet
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
        return np.zeros( x0.shape ), 0
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

def generateSamplingMask(imsz, p, saType='bernoulli', num_patterns=1, seed=1234321):
  # p is the undersampling ratio: what you don't sample
  if(p < 0.0)or(p > 1.0):
    print('ERROR: Invalid undersampling ratio delta in generateSamplingMask.')
    sys.exit(-1)
  elif(p == 0.0):
    return np.full(imsz, True, dtype=bool)
  else:
    np.random.seed(seed)
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
