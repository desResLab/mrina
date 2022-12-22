# %% 
import numpy as np
import numpy.fft as fft
import numpy.linalg as la
import pywt

# Generic Operator
class genericOperator(object):
    pass

# Implementation of a simple linear operator
class OperatorLinear(genericOperator):
  def __init__(self, mat, samplingSet=None, basisSet=None):
    self.__mat = mat
    self.__shape = mat.shape

    # Define input and output shape for post-multiplication
    self.inShape = self.__mat.shape[1]
    self.outShape = self.__mat.shape[0]

    # Define input and output shape for post-multiplication
    self.wavShape = self.__mat.shape[1]
    self.imShape = self.__mat.shape[0]

    # Assign sampling set
    if isinstance(samplingSet, list):
      samplingSet = np.array(samplingSet)
    tmp = np.zeros(self.outShape,dtype=bool)
    tmp[samplingSet] = True
    self.samplingSet = tmp

    # Assign basis set
    if isinstance(basisSet, list):
      basisSet = np.array(basisSet)
    tmp = np.zeros(self.inShape,dtype=bool)
    tmp[basisSet] = True
    self.basisSet = tmp

  def eval(self, x, mode=1):
    if(mode==1):
      
      # Direct map

      if(self.basisSet is not None):
        tmp = np.dot(self.__mat[:,self.basisSet],x[self.basisSet])
      else: 
        tmp = np.dot(self.__mat,x)

      if(self.samplingSet is not None):
        return tmp[self.samplingSet]
      else:
        return tmp

    elif(mode==2):
      
      # Adjoint map

      if(self.samplingSet is not None):
        tmp = np.dot(np.conjugate(self.__mat.T[:,self.samplingSet]),x[self.samplingSet])
      else:
        tmp = np.dot(np.conjugate(self.__mat.T),x)

      if(self.basisSet is not None):
        return tmp[self.basisSet]
      else: 
        return tmp

  def adjoint(self, x):
      return self.eval(x, mode=2)

  @property
  def shape(self):
    "The shape of the operator."
    return self.__shape

  def input_size(self):
    return self.__shape[1]

  @property
  def T(self):
    "Transposed of the operator."
    return OperatorLinear(np.conjugate(self.__mat.T))

  def __matmul__(self, x):
    return np.dot(self.__mat,x)

  def colRestrict(self,idxSet=None):
    return OperatorLinear(self.__mat[:,idxSet])

  def norm(self):
    return np.linalg.norm(self.__mat,2)

# %%
# Find the name of the wavelet associated to the adjoint of
# the wavelet transform
def getAdjointWavelet(waveletName):
    if waveletName is None or waveletName == 'None':
        return None
    if len(waveletName) > 4 and waveletName[0:4] == 'bior':
        return 'rbio' + waveletName[4::]
    if len(waveletName) > 4 and waveletName[0:4] == 'rbio':
        return 'bior' + waveletName[4::]
    return waveletName

def getWaveletTransformShape(imShape, waveletName, waveletLevel=None):
    if waveletName is None or waveletName == 'None':
        return imShape

    x = np.zeros(imShape)
    Wx = pywt.wavedec2(x, waveletName, mode='zero', level=waveletLevel)

    return pywt.coeffs_to_array(Wx)[0].shape

def getWaveletTransformSlices(imShape, waveletName, waveletLevel=None):
    if waveletName is None or waveletName == 'None':
        return None

    x = np.zeros(imShape)
    Wx = pywt.wavedec2(x, waveletName, mode='zero', level=waveletLevel)

    return pywt.coeffs_to_array(Wx)[1]

def getWaveletReconstructionShape(imShape, waveletName, waveletLevel=None):
    if waveletName is None or waveletName == 'None':
        return imShape

    Wx = pywt.wavedec2(np.zeros(imShape), waveletName, mode='zero', level=waveletLevel)

    return pywt.waverec2(Wx, wavelet=waveletName, mode='zero').shape

# OperatorWaveletToFourier
#   This implements the linear map that takes as inputs
#   the wavelet coefficients of a complex image, and 
#   outputs the Fourier coefficients of the image. The 
#   map supports restricting the support of the wavelet
#   coefficients, and subsampling the Fourier coefficients
class OperatorWaveletToFourier(genericOperator):
    def __init__(self, imShape, samplingSet=None, basisSet=None, isTransposed=False, waveletName=None, waveletLevel=None):
        # Check for boundary case
        if waveletName == 'None':
            waveletName = None
        # Parameters
        self.imShape = imShape
        self.waveletName = waveletName
        self.waveletNameAdj = getAdjointWavelet(waveletName)
        self.waveletLevel = waveletLevel
        self.wavShape = getWaveletTransformShape(imShape, waveletName, waveletLevel)
        self.wavSlices = getWaveletTransformSlices(imShape, waveletName, waveletLevel)
        self.isTransposed = isTransposed
        self._norm = None
        # Test if the image shape produces a consistent reconstruction
        # using pyWavelets or not
        xrecShape = getWaveletReconstructionShape(imShape, waveletName, waveletLevel)
        self.waveletCrop = [ xrecShape[0] != imShape[0], 
                             xrecShape[1] != imShape[1] ]
        # Validate shapes
        if samplingSet is not None:

            # The sampling set should represent a 2D complex image
            if isinstance(samplingSet, list):
                samplingSet = np.array(samplingSet)
            if(samplingSet.ndim < 2):
                tmp = np.zeros(self.imShape,dtype=bool).flatten()
                tmp[samplingSet] = True
                samplingSet = tmp.reshape(self.imShape)
            # Check if the size is correct
            if samplingSet.shape[0] != self.imShape[0] or samplingSet.shape[1] != self.imShape[1]:
                raise ValueError('The sampling array does not match the shape of the image.')

        if basisSet is not None:

            # If the basisSet is given in terms of an indexset and not a binary mask, convert it to a binary mask
            # Basis Set should refer to a 2D Wavelet coefficient representation
            if isinstance(basisSet, list):
                basisSet = np.array(basisSet)
            if(basisSet.ndim < 2):
                tmp = np.zeros(self.wavShape,dtype=bool).flatten()
                tmp[basisSet] = True
                basisSet = tmp.reshape(self.wavShape)
            # Check if the size is correct
            if basisSet.shape[0] != self.wavShape[0] or basisSet.shape[1] != self.wavShape[1]:
                raise ValueError('The basis indices do not match the shape of the wavelet transform.') 

        # Restriction of the support of the wavelet coefficients
        self.basisSet = basisSet
        if basisSet is None:
            basisShape = self.wavShape
        else:
            basisShape = (np.count_nonzero(basisSet),)
        # Subsampling of Fourier coefficients
        self.samplingSet = samplingSet
        if samplingSet is None:
            samplingShape = self.imShape
        else:
            samplingShape = (np.count_nonzero(samplingSet),)
        # Input and output shapes
        if isTransposed:
            self.inShape = samplingShape
            self.outShape = basisShape
        else:
            self.inShape = basisShape
            self.outShape = samplingShape

    # The method eval is the only one that should use the self.isTransposed flag.
    def eval(self, x, mode=1):
        # print("shape of x: ",x.shape)
        # Verify if the instance is transposed
        if self.isTransposed:
            if mode == 1:
                mode = 2
            else:
                mode = 1
        # Evaluate forward map
        if mode == 1:

            # Check input dimension
            if(x.shape != self.wavShape):
              raise ValueError('ERROR: Input for direct application of the operator has not the correct size.')

            # Verify if the support of the wavelet transform is restricted
            if self.waveletName is None:
                if self.basisSet is None:
                    _x = x
                else:
                    _x = np.zeros(self.wavShape, dtype=complex)
                    _x[self.basisSet] = x[self.basisSet]
            else:
                if self.basisSet is None:
                    _w = pywt.array_to_coeffs(x, self.wavSlices, output_format='wavedec2')
                else:
                    # Remove the wavelet coefficients not in the basis set
                    x[np.logical_not(self.basisSet)] = 0.0                    
                    _w = pywt.array_to_coeffs(x, self.wavSlices, output_format='wavedec2')
                # Compute image from wavelet coefficients
                _x = pywt.waverec2(_w, wavelet=self.waveletName, mode='zero')
            # Verify if reconstruction is consistent
            if self.waveletCrop[0]:
                _x = _x[0:self.imShape[0], :]
            if self.waveletCrop[1]:
                _x = _x[:, 0:self.imShape[1]]
            # Verify if there is subsampling of the Fourier coefficients
            if self.samplingSet is None:
                return fft.fft2(_x, norm='ortho')
            else:
                _f = fft.fft2(_x, norm='ortho')
                _f[np.logical_not(self.samplingSet)] = 0.0 # Set the Fourier coefficients outside the set to zero
                # return _f[self.samplingSet]
                return _f

        # Evaluate adjoint
        if mode == 2:

            # Check input dimension
            if(x.shape != self.imShape):
              raise ValueError('ERROR: Input for inverse application of the operator has not the correct size.')

            # Verify if there is subsampling of the Fourier coefficients
            if self.samplingSet is None:
                _im = fft.ifft2(x, norm='ortho')
            else:
                _f = np.zeros(self.imShape, dtype=complex)
                _f[self.samplingSet] = x[self.samplingSet]
                # _f = x
                # _f[np.logical_not(self.samplingSet)] = 0.0
                _im = fft.ifft2(_f, norm='ortho')
            if self.waveletName is None:
                _w = _im
            else:
                _w = pywt.wavedec2(_im, wavelet=self.waveletNameAdj, mode='zero', level=self.waveletLevel)
                _w = pywt.coeffs_to_array(_w)[0]
            # Verify if the support of the wavelet transform is restricted
            if self.basisSet is None:
                return _w
            else:
                _w[np.logical_not(self.basisSet)] = 0.0
                # return _w[self.basisSet]
                return _w

    def adjoint(self, x):
        return self.eval(x, mode=2)

    def norm(self, maxItns=1E3, absTol=1E-6, relTol=1E-9):
        if self._norm is not None:
            return self._norm
        # Initialize variables
        x = np.random.normal(size=self.wavShape) + 1j * np.random.normal(size=self.wavShape)
        x = x / la.norm(x)
        s = 0
        ds = np.inf
        itn = 0
        stop = False
        # Power iteration loop
        while not stop:
            # Evaluate xp = A'A x
            xp = self.adjoint(self.eval(x))
            # Evaluate x' A'Ax to estimate sigma_max^2
            sp = np.sqrt(np.real(np.sum(np.conj(x.flatten()) * xp.flatten())))
            ds = np.abs(sp - s)
            if ds < absTol or ds < absTol * s or itn > maxItns:
                stop = True
            # Normalize singular vector
            x  = xp / la.norm(xp)
            s  = sp
        return s

    def getImageFromWavelet(self, x):
        if(x.shape != self.wavShape):
            raise ValueError('ERROR: Invalid input size for getImageFromWavelet.')
        if self.waveletName is None:
            if self.basisSet is None:
                _x = x
            else:
                _x = np.zeros(self.imShape, dtype=complex)
                _x[self.basisSet] = x[:]
        else:
            if self.basisSet is None:
                _w = pywt.array_to_coeffs(x, self.wavSlices, output_format='wavedec2')
            else:
                # Set to zero wavelet coefficients not in the basis set 
                x[np.logical_not(self.basisSet)] = 0.0
                _w = pywt.array_to_coeffs(x, self.wavSlices, output_format='wavedec2')
            # Compute image from wavelet coefficients
            _x = pywt.waverec2(_w, wavelet=self.waveletName, mode='zero')
        # Verify if reconstruction is consistent
        if self.waveletCrop[0]:
            _x = _x[0:self.imShape[0], :]
        if self.waveletCrop[1]:
            _x = _x[:, 0:self.imShape[1]]
        return _x

    def getImageFromFourier(self, y):
        if self.samplingSet is None:
            _im = fft.ifft2(y, norm='ortho')
        else:
            y[np.logical_not(self.samplingSet)] = 0.0
            _im = fft.ifft2(y, norm='ortho')
        return _im

    @property
    def shape(self):
        # This is the shape of the matrix representing
        # the operator with vectorized inputs and outputs
        return (np.prod(self.outShape), np.prod(self.inShape))

    def __matmul__(self, x):
        _y = self.eval(np.reshape(x, newshape=self.inShape))
        return _y.ravel()
        
    @property
    def T(self):
        # Instantiate the transpose of the operator
        return OperatorWaveletToFourier(imShape=self.imShape, samplingSet=self.samplingSet, basisSet=self.basisSet, isTransposed=not(self.isTransposed), waveletName=self.waveletName, waveletLevel=self.waveletLevel)

    # Create An operator with a 
    def colRestrict(self, basisSet=None):
        # Instantiate operator restricted to some entries
        return OperatorWaveletToFourier(imShape=self.imShape, samplingSet=self.samplingSet, basisSet=basisSet, isTransposed=self.isTransposed, waveletName=self.waveletName, waveletLevel=self.waveletLevel)

    # Print Operator
    def __str__(self):
        res  = '--- Wavelet to Fourier Operator\n'
        res += 'Shape of the original image: %d x %d\n' % (self.imShape[0],self.imShape[1])
        res += 'Wavelet name: %s\n' % (self.waveletName)
        res += 'Wavelet adjoint name: %s\n' % (self.waveletNameAdj)
        if self.waveletLevel is not None:
            res += 'Wavelet level: %d\n' % (self.waveletLevel)
        else:
            res += 'Wavelet level is not defined\n'
        res += 'Shape of the wavelet transform: %d x %d\n' % (self.wavShape[0],self.wavShape[1])
        res += 'Is operator transposed: ' + str(self.isTransposed) + '\n'
        # Sampling and basis sets
        if self.samplingSet is not None:
            res += 'Sampling set has size: %d x %d\n' % (self.samplingSet.shape[0],self.samplingSet.shape[1])
        else:
            res += 'Sampling set is not defined\n'
        if self.basisSet is not None:
            res += 'Basis set has size: %d x %d\n' % (self.basisSet.shape[0],self.basisSet.shape[1])
        else:
            res += 'Basis set is not defined\n'
        # Input and output shapes
        res += 'Input shape is: ' + str(self.inShape) + '\n'
        res += 'Output shape is: ' + str(self.outShape) + '\n'
        return res

# OperatorWaveletToFourierX4
#   This implements the linear map that takes as inputs
#   the wavelet coefficients of 4 complex images, and 
#   outputs their Fourier coefficients. The map supports restricting 
#   the support of the wavelet coefficients, and subsampling
#   the Fourier coefficients
class OperatorWaveletToFourierX4(genericOperator):
    def __init__(self, imShape, samplingSet=None, basisSet=None, isTransposed=False, waveletName='haar', waveletLevel=None):
        # Parameters
        self.imShape = imShape
        self.waveletName = waveletName
        self.waveletNameAdj = getAdjointWavelet(waveletName)
        self.waveletLevel = waveletLevel
        self.wavShape = getWaveletTransformShape(imShape, waveletName, waveletLevel)
        self.wavSlices = getWaveletTransformSlices(imShape, waveletName, waveletLevel)
        self.isTransposed = isTransposed
        self.samplingSet = samplingSet
        self.basisSet = basisSet
        self._norm = None
        # Generate array of maps
        if samplingSet is None and basisSet is None:
            self.map = [ OperatorWaveletToFourier(imShape=imShape, samplingSet=None, basisSet=None, isTransposed=isTransposed, waveletName=waveletName, waveletLevel=waveletLevel) for I in range(4) ]
        else:
            if samplingSet is None:
                if basisSet.ndim == 3:
                    self.map = [ OperatorWaveletToFourier(imShape=imShape, samplingSet=None, basisSet=basisSet[:, :, I], isTransposed=isTransposed, waveletName=waveletName, waveletLevel=waveletLevel) for I in range(4) ]
                else:
                    self.map = [ OperatorWaveletToFourier(imShape=imShape, samplingSet=None, basisSet=basisSet, isTransposed=isTransposed, waveletName=waveletName, waveletLevel=waveletLevel) for I in range(4) ]
            else:
                if samplingSet.ndim == 3:
                    if basisSet is None:
                        self.map = [ OperatorWaveletToFourier(imShape=imShape, samplingSet=samplingSet[:, :, I], basisSet=None, isTransposed=isTransposed, waveletName=waveletName, waveletLevel=waveletLevel) for I in range(4) ]
                    else:
                        if basisSet.ndim == 3:
                            self.map = [ OperatorWaveletToFourier(imShape=imShape, samplingSet=samplingSet[:, :, I], basisSet=basisSet[:, :, I], isTransposed=isTransposed, waveletName=waveletName, waveletLevel=waveletLevel) for I in range(4) ]
                        else:
                            self.map = [ OperatorWaveletToFourier(imShape=imShape, samplingSet=samplingSet[:, :, I], basisSet=basisSet, isTransposed=isTransposed, waveletName=waveletName, waveletLevel=waveletLevel) for I in range(4) ]
                else:
                    if basisSet is None:
                        self.map = [ OperatorWaveletToFourier(imShape=imShape, samplingSet=samplingSet, basisSet=None, isTransposed=isTransposed, waveletName=waveletName, waveletLevel=waveletLevel) for I in range(4) ]
                    else:
                        if basisSet.ndim == 3:
                            self.map = [ OperatorWaveletToFourier(imShape=imShape, samplingSet=samplingSet, basisSet=basisSet[:, :, I], isTransposed=isTransposed, waveletName=waveletName, waveletLevel=waveletLevel) for I in range(4) ]
                        else:
                            self.map = [ OperatorWaveletToFourier(imShape=imShape, samplingSet=samplingSet, basisSet=basisSet, isTransposed=isTransposed, waveletName=waveletName, waveletLevel=waveletLevel) for I in range(4) ]
        # Find input shape and output shapes
        inShape = [ self.map[I].inShape for I in range(4) ]
        self.inSlices = None
        if len(inShape[0]) == 2:
            self.inShape = (inShape[0][0], inShape[0][1], 4)
        else:
            self.inShape = np.prod(inShape[0])
            self.inSlices = [ [0, 0] for I in range(4) ]
            self.inSlices[0][1] = inShape[0][0]
            for I in range(1, 4):
                self.inSlices[I][0] = self.inSlices[I-1][1]
                self.inSlices[I][1] = self.inSlices[I][0] + inShape[I][0]
                self.inShape = self.inShape + np.prod(inShape[I])
            self.inShape = (self.inShape,)
        self._inShape = self.inShape
        
        outShape = [ self.map[I].outShape for I in range(4) ]
        self.outSlices = None
        if len(outShape[0])== 2:
            self.outShape = (outShape[0][0], outShape[0][1], 4)
        else:
            self.outShape = np.prod(outShape[0])
            self.outSlices = [ [0, 0] for I in range(4) ]
            self.outSlices[0][1] = outShape[0][0]
            for I in range(1, 4):
                self.outSlices[I][0] = self.outSlices[I-1][1]
                self.outSlices[I][1] = self.outSlices[I][0] + outShape[I][0]
                self.outShape = self.outShape + np.prod(outShape[I])
            self.outShape = (self.outShape,)
        self._outShape = self.outShape

    def eval(self, x, mode=1):
        if mode == 1:
            _y = np.zeros(self._outShape, dtype=complex)
            if self.inSlices is None:
                if self.outSlices is None:
                    for I in range(4):
                        _y[:, :, I] = self.map[I].eval(x[:, :, I], mode)
                else:
                    for I in range(4):
                        _y[self.outSlices[I][0]:self.outSlices[I][1]] = self.map[I].eval(x[:, :, I], mode)
            else:
                if self.outSlices is None:
                    for I in range(4):
                        _y[:, :, I] = self.map[I].eval(x[self.inSlices[I][0]:self.inSlices[I][1]], mode)
                else:
                    for I in range(4):
                        _y[self.outSlices[I][0]:self.outSlices[I][1]] = self.map[I].eval(x[self.inSlices[I][0]:self.inSlices[I][1]], mode)
            return _y
        if mode == 2:
            _x = np.zeros(self._inShape, dtype=complex)
            if self.outSlices is None:
                if self.inSlices is None:
                    for I in range(4):
                        _x[:, :, I] = self.map[I].eval(x[:, :, I], mode)
                else:
                    for I in range(4):
                        _x[self.inSlices[I][0]:self.inSlices[I][1]] = self.map[I].eval(x[:, :, I], mode)
            else:
                if self.inSlices is None:
                    for I in range(4):
                        _x[:, :, I] = self.map[I].eval(x[self.outSlices[I][0]:self.outSlices[I][1]], mode)
                else:
                    for I in range(4):
                        _x[self.inSlices[I][0]:self.inSlices[I][1]] = self.map[I].eval(x[self.outSlices[I][0]:self.outSlices[I][1]], mode)
            return _x

    def adjoint(self, x):
        return self.eval(x, mode=2)

    def norm(self, maxItns=1E3, absTol=1E-6, relTol=1E-9):
        if self._norm is not None:
            return self._norm        
        # Initialize variables
        x = np.random.normal(size=self._inShape) + 1j * np.random.normal(size=self._inShape)
        x = x / la.norm(x)
        s = 0
        ds = np.inf
        itn = 0
        stop = False
        # Power iteration loop
        while not stop:
            # Evaluate xp = A'A x
            xp = self.adjoint(self.eval(x))
            # Evaluate x' A'Ax to estimate sigma_max^2
            sp = np.sqrt(np.real(np.sum(np.conj(x.flatten()) * xp.flatten())))
            ds = np.abs(sp - s)
            if ds < absTol or ds < absTol * s or itn > maxItns:
                stop = True
            # Normalize singular vector
            x  = xp / la.norm(xp)
            s  = sp
        return s

    def getImageFromWavelet(self, x):
        _im = np.zeros(self.imShape + (4,), dtype=complex)
        if self.inSlices is None:
            for I in range(4):
                _im[:, :, I] = self.map[I].getImageFromWavelet(x[:, :, I])
        else:
            for I in range(4):
                _im[:, :, I] = self.map[I].getImageFromWavelet(x[self.inSlices[I][0]:self.inSlices[I][1]])
        return _im

    def getImageFromFourier(self, y):
        _im = np.zeros(self.imShape + (4,), dtype=complex)
        if self.outSlices is None:
            for I in range(4):
                _im[:, :, I] = self.map[I].getImageFromFourier(y[:, :, I])
        else:
            for I in range(4):
                _im[:, :, I] = self.map[I].getImageFromFourier(y[self.outSlices[I][0]:self.outSlices[I][1]])
        return _im

    @property
    def shape(self):
        # This is the shape of the matrix representing
        # the operator with vectorized inputs and outputs
        return (np.prod(self._outShape), np.prod(self._inShape))

    def __matmul__(self, x):
        _y = self.eval(np.reshape(x, newshape=self._inShape))
        return _y.ravel()
        
    @property
    def T(self):
        # Instantiate the transpose of the operator
        return OperatorWaveletToFourierX4(imShape=self.imShape, samplingSet=self.samplingSet, basisSet=self.basisSet, isTransposed=not(self.isTransposed), waveletName=self.waveletName, waveletLevel=self.waveletLevel)

    # Create An operator with a 
    def colRestrict(self, basisSet=None):
        # Instantiate operator restricted to some entries
        return OperatorWaveletToFourierX4(imShape=self.imShape, samplingSet=self.samplingSet, basisSet=basisSet, isTransposed=self.isTransposed, waveletName=self.waveletName, waveletLevel=self.waveletLevel)


# OperatorFourierLowRank
#   This implements the linear map that takes as inputs
#   the wavelet coefficients of 4 complex images, and 
#   outputs their Fourier coefficients. The map supports restricting 
#   the support of the wavelet coefficients, and subsampling
#   the Fourier coefficients
class OperatorFourierLowRank(OperatorWaveletToFourierX4):
    def __init__(self, imShape, samplingSet=None, isTransposed=False):
        super().__init__(imShape, samplingSet=samplingSet, basisSet=None, isTransposed=isTransposed, waveletName=None, waveletLevel=None)
        self.mtxShape = (np.prod(imShape), 4)
        self.arrShape = (imShape[0], imShape[1], 4)
        if self.isTransposed:
            self.outShape = (np.prod(imShape), 4)
        else:
            self.inShape = (np.prod(imShape), 4)

    def eval(self, x, mode=1):
        if self.isTransposed:
            if mode == 2:
                return super().eval(np.reshape(x, newshape=self.arrShape), mode)
            else:
                return np.reshape(super().eval(x, mode), newshape=self.mtxShape)
                
        else:
            if mode == 1:
                return super().eval(np.reshape(x, newshape=self.arrShape), mode)
            else:
                return np.reshape(super().eval(x, mode), newshape=self.mtxShape)

    def __matmul__(self, x):
        _y = self.eval(np.reshape(x, newshape=self.inShape))
        return _y.ravel()

    @property
    def T(self):
        # Instantiate the transpose of the operator
        return OperatorFourierLowRank(imShape=self.imShape, samplingSet=self.samplingSet, isTransposed=not(self.isTransposed))

    # Create An operator with a 
    def colRestrict(self, basisSet=None):
        # Instantiate operator restricted to some entries
        raise NotImplementedError('colRestrict is not implemented for OperatorFourierLowRank.')

# %%
def testLinearMap(A, ns=10):
    err_mean = 0.0
    err_std = 0.0
    err_min = np.inf
    err_max = -np.inf
    err_mtx_eval = -np.inf
    err_mtx_adj = -np.inf
    err_adj_eval = -np.inf
    err_adj_adj = -np.inf
    At = A.T
    for I in range(ns):
        x = np.random.normal(size=A.inShape) + 1j * np.random.normal(size=A.inShape)
        y = np.random.normal(size=A.outShape) + 1j * np.random.normal(size=A.outShape)
        Ax = A.eval(x)
        _Ax = At.adjoint(x)
        tAy = A.adjoint(y)
        _tAy = At.eval(y)
        yAx = np.sum( np.conj(y.ravel()) * Ax.ravel() )
        tAyx = np.sum( np.conj(tAy.ravel()) * x.ravel() )
        err = np.abs(yAx - tAyx)
        err_mean = err
        err_std = err ** 2
        err_max = np.maximum(err_max, err)
        err_min = np.minimum(err_min, err)
        err_mtx_eval = np.maximum(err_mtx_eval, la.norm(A @ x.ravel() - Ax.ravel()))
        err_mtx_adj = np.maximum(err_mtx_adj, la.norm(A.T @ y - tAy.ravel()))
        err_adj_eval = np.maximum(err_mtx_eval, la.norm(Ax.ravel() - _Ax.ravel()))
        err_adj_adj = np.maximum(err_mtx_adj, la.norm(_tAy.ravel() - tAy.ravel()))
    err_mean = err_mean / ns
    err_std = np.sqrt(err_std / ns - err_mean ** 2)
    return err_mean, err_std, err_min, err_max, err_mtx_eval, err_mtx_adj, err_adj_eval, err_adj_adj

if __name__ == '__main__':
    ns = 10
    do_sampling = False
    imShape = [ (71, 77), (128, 128) ]
    waveletName = [ 'None', 'haar', 'db4', 'sym5', 'coif5', 'dmey', 'bior2.6', 'rbio2.8', 'dmey' ]
    for _imShape in imShape:
        for _waveletName in waveletName:
            print('-------------------------------------------------------')
            print('Testing operator for image size {:d} x {:d} and wavelet {:s}'.format(_imShape[0], _imShape[1], _waveletName))
            print('-------------------------------------------------------')
            print('\n ****** OperatorWaveletToFourier ***********************')
            if do_sampling:
                delta = np.random.uniform()
                rho = np.random.uniform()
                print('Sampling set ratio (fraction kept):  {:1.3f}'. format(delta))
                print('Basis set ratio (fraction kept):     {:1.3f}'. format(rho))
                samplingSet = np.where(np.random.uniform(size=_imShape) < delta, True, False)
                basisSet = np.where(np.random.uniform(size=getWaveletTransformShape(_imShape, _waveletName)) < rho, True, False)
                A = OperatorWaveletToFourier(_imShape, samplingSet=samplingSet, basisSet=basisSet, isTransposed=False, waveletName=_waveletName)
                inShape = A.inShape
                outShape = A.outShape

                print('     Input shape:   {:d} x 1'.format(inShape[0]))
                print('     Output shape:  {:d} x 1'.format(outShape[0]))
            else:
                A = OperatorWaveletToFourier(_imShape, samplingSet=None, basisSet=None, isTransposed=False, waveletName=_waveletName)
                inShape = A.inShape
                outShape = A.outShape

                print('     Input shape:   {:d} x {:d}'.format(inShape[0], inShape[1]))
                print('     Output shape:  {:d} x {:d}'.format(outShape[0], outShape[1]))

            print('     Operator norm: {:1.5E}'.format(A.norm()))
            print('Testing inner products for {:d} samples...'.format(ns))
            err_mean, err_std, err_min, err_max, err_mtx_eval, err_mtx_adj, err_adj_eval, err_adj_adj = testLinearMap(A)
            print('     Max. Error : {:1.5E}'.format(err_max))
            print('     Min. Error : {:1.5E}'.format(err_min))
            print('     Mean Error : {:1.5E}'.format(err_mean))
            print('          Std   : {:1.5E}'.format(err_std))
            print('     Implementation of __matmul__')
            print('         Max. Error (eval) : {:1.5E}'.format(err_mtx_eval))
            print('         Min. Error (adj)  : {:1.5E}'.format(err_mtx_adj))
            print('     Implementation of .T')
            print('         Max. Error (eval) : {:1.5E}'.format(err_adj_eval))
            print('         Min. Error (adj)  : {:1.5E}'.format(err_adj_adj))

            print('\n ****** OperatorWaveletToFourierX4 *********************')
            if do_sampling:
                delta = np.random.uniform()
                rho = np.random.uniform()
                print('Sampling set ratio (fraction kept):  {:1.3f}'. format(delta))
                print('Basis set ratio (fraction kept):     {:1.3f}'. format(rho))
                samplingSet = np.where(np.random.uniform(size=_imShape) < delta, True, False)
                basisSet = np.where(np.random.uniform(size=getWaveletTransformShape(_imShape, _waveletName)) < rho, True, False)
                A = OperatorWaveletToFourierX4(_imShape, samplingSet=samplingSet, basisSet=basisSet, isTransposed=False, waveletName=_waveletName)
                inShape = A.inShape
                outShape = A.outShape

                print('     Input shape:   {:d} x 1'.format(inShape[0]))
                print('     Output shape:  {:d} x 1'.format(outShape[0]))
            else:
                A = OperatorWaveletToFourierX4(_imShape, samplingSet=None, basisSet=None, isTransposed=False, waveletName=_waveletName)
                inShape = A.inShape
                outShape = A.outShape

                print('     Input shape:   {:d} x {:d}'.format(inShape[0], inShape[1]))
                print('     Output shape:  {:d} x {:d}'.format(outShape[0], outShape[1]))

            print('     Operator norm: {:1.5E}'.format(A.norm()))
            print('Testing inner products for {:d} samples...'.format(ns))
            err_mean, err_std, err_min, err_max, err_mtx_eval, err_mtx_adj, err_adj_eval, err_adj_adj = testLinearMap(A)
            print('     Max. Error : {:1.5E}'.format(err_max))
            print('     Min. Error : {:1.5E}'.format(err_min))
            print('     Mean Error : {:1.5E}'.format(err_mean))
            print('          Std   : {:1.5E}'.format(err_std))
            print('     Implementation of __matmul__')
            print('         Max. Error (eval) : {:1.5E}'.format(err_mtx_eval))
            print('         Min. Error (adj)  : {:1.5E}'.format(err_mtx_adj))
            print('     Implementation of .T')
            print('         Max. Error (eval) : {:1.5E}'.format(err_adj_eval))
            print('         Min. Error (adj)  : {:1.5E}'.format(err_adj_adj))

            print('\n ****** OperatorFourierLowRank *************************')
            if do_sampling:
                delta = np.random.uniform()
                rho = np.random.uniform()
                print('Sampling set ratio (fraction kept):  {:1.3f}'. format(delta))
                # print('Basis set ratio (fraction kept):     {:1.3f}'. format(rho))
                samplingSet = np.where(np.random.uniform(size=_imShape) < delta, True, False)
                # basisSet = np.where(np.random.uniform(size=getWaveletTransformShape(_imShape, _waveletName)) < rho, True, False)
                A = OperatorFourierLowRank(_imShape, samplingSet=samplingSet, isTransposed=False)
                inShape = A.inShape
                outShape = A.outShape

                print('     Input shape:   {:d} x 1'.format(inShape[0]))
                print('     Output shape:  {:d} x 1'.format(outShape[0]))
            else:
                A = OperatorFourierLowRank(_imShape, samplingSet=None, isTransposed=False)
                inShape = A.inShape
                outShape = A.outShape

                print('     Input shape:   {:d} x {:d}'.format(inShape[0], inShape[1]))
                print('     Output shape:  {:d} x {:d}'.format(outShape[0], outShape[1]))

            print('     Operator norm: {:1.5E}'.format(A.norm()))
            print('Testing inner products for {:d} samples...'.format(ns))
            err_mean, err_std, err_min, err_max, err_mtx_eval, err_mtx_adj, err_adj_eval, err_adj_adj = testLinearMap(A)
            print('     Max. Error : {:1.5E}'.format(err_max))
            print('     Min. Error : {:1.5E}'.format(err_min))
            print('     Mean Error : {:1.5E}'.format(err_mean))
            print('          Std   : {:1.5E}'.format(err_std))
            print('     Implementation of __matmul__')
            print('         Max. Error (eval) : {:1.5E}'.format(err_mtx_eval))
            print('         Min. Error (adj)  : {:1.5E}'.format(err_mtx_adj))
            print('     Implementation of .T')
            print('         Max. Error (eval) : {:1.5E}'.format(err_adj_eval))
            print('         Min. Error (adj)  : {:1.5E}'.format(err_adj_adj))            

# %%
