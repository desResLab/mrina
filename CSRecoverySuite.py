import numpy.fft as fft
import pywt
import numpy as np
import sys
import numpy.linalg as la
import math
import time
#import sklearn.linear_model as lasso#for comparison?
#reg = lasso.LassoLars(lambda)
#reg.fit(trainX, trainY)

#methods for cropping in the case the dimensions aren't a power of 2
def powerof2(num):
    #return the highest power of 2 less than or equal to number
    return int(math.pow(2,math.floor(math.log(num, 2))))

def crop(x):
    if x.shape[1] > x.shape[0]:
        dim1= powerof2(x.shape[0])
        #max between power of 2 or multiple of other dimension
        dim2 = max(powerof2(x.shape[1]), math.floor(x.shape[1]/dim1)*dim1)
    else:
        dim2= powerof2(x.shape[1])
        dim1 = max(powerof2(x.shape[0]), math.floor(x.shape[0]/dim2)*dim2)
    return x[0:dim1, 0:dim2]

def pywt2array( x, shape ):
    #essentially a breadth-first traversal
    y = np.zeros((0,1))
    for sublist in x:
        for z in sublist:
            y = np.concatenate((y, np.expand_dims(z.ravel(), axis=1)),axis=0)
    y = y.reshape(shape)
    return y

def array2pywt( x ):
    shape = x.shape
    #assuming both dims power of 2 or one dim is multiple of other (which is power of 2)
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

def sft( x, t ):
    # Soft-thresholding
    return x * ( 1 - np.minimum( t / np.maximum( np.abs(x), 1E-32 ), 1 ) )

def project_l1_ball( x, t, dxAbsTol=1E-9, fxAbsTol=1E-12 ):
    xL1Nrm  = la.norm( x.ravel(), 1 );
    if xL1Nrm > t:
        # If the input is outside the l1-ball we project.
        nx      = x.size;
        # Sort magnitudes in decreasing order
        mx      = np.flip( np.sort( np.absolute( x.ravel() ), kind='mergesort' ) );
        # Obtain unique values (this is needed for some borderline cases)
        umx     = np.flip( np.unique( mx ) );
        if( t <= umx[0] - umx[1] ):
            # If the radius of the l1-ball is small and the input is far away
            # and the shrinkage is large.
            vp      = umx[0];
            sroot   = vp - t;
            return sft( x, sroot )
        else:
            # This is the average case
            cmx     = np.cumsum( mx );
            smx     = cmx - np.array( range(1, nx+1) ) * mx;
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

def bisect( f, a, b, maxItns=1E3, dxAbsTol=1E-6, fxAbsTol=1E-12 ):
    itn     = 0;
    # assuming for now decreasing fcn s.t. a has positive f(a), and b negative f(b)
    fa      = f( a );
    fb      = f( b );
    dx      = b - a;
    if abs(fa) < fxAbsTol:
        return a
    elif abs(fb) < fxAbsTol:
        return b
    if fa < 0 or fb > 0:
        print('exit')
        sys.exit(-1)
    while( dx > dxAbsTol and itn < maxItns ):
        itn     = itn + 1;
        midpt   = 0.5 * (a + b);
        fmid    = f( midpt );
        if( np.abs( fmid ) < fxAbsTol ):
            return midpt
        if f( midpt ) > 0:
            a       = midpt
            fa      = fmid
        else:
            b       = midpt
            fb      = fmid
        dx      = b-a;
    return midpt

# Defines a class for linear transforms
class Operator4dFlow(object):
    def __init__(self, insz=None, imsz=None, samplingSet=None, waveletName='haar', waveletMode='periodic'):
        # insz is the size of the array used as input
        self.insz           = insz;
        self.imsz           = imsz;
        self.waveletName    = waveletName;
        self.waveletMode    = waveletMode;
        # samplingSet is the set of indices that are measured after the mapping
        #             has been applied
        self.samplingSet    = samplingSet;
        if( samplingSet is None ):
            self.outsz          = imsz;
            self._buffer        = None;
        else:
            self.outsz          = ( np.count_nonzero( samplingSet ), );
            self._buffer        = np.zeros( self.imsz ) + 1j * np.zeros( self.imsz );
        self._cst           = math.pow( np.prod( insz ), -1/2 );
    #  The method eval implements
    #   obj.eval(x, 1) returns A * x
    #   obj.eval(x, 2) returns A' * x where A' is the adjoint
    def eval(self, x, mode):
        if( mode == 1 ): # the forward map
            if( self.samplingSet is None ):
                return self._cst * fft.fft2(pywt.waverec2(array2pywt(x), wavelet=self.waveletName, mode=self.waveletMode));
            else:
                y = self._cst * fft.fft2(pywt.waverec2(array2pywt(x), wavelet=self.waveletName, mode=self.waveletMode));
                return y[ self.samplingSet ];
        if( mode == 2 ): # the adjoint map
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
        return self.insz;

def f( y, A, alpha ):
  return 0.5 * math.pow(la.norm((y - A.eval(alpha, 1)).ravel(), 2), 2)

def f_grad( y, A, alpha ):
  return A.eval(A.eval(alpha, 1) - y, 2);

# Computes leading singular value of the linear mapping A via power iteration
def OperatorNorm( A, maxItns=1E3, dsAbsTol=1E-6 ):
    # Initialize variables
    b       = np.random.normal( size = A.input_size() );
    b       = b / la.norm(b)
    s       = 0;
    ds      = np.inf;
    itn     = 0;
    # Power iteration loop
    while( ds > dsAbsTol and itn <= maxItns ):
        # Evaluate xk = A'A x
        bk      = A.eval(A.eval(b, 1), 2);
        # Evaluate xk' A'Ax to estimate sigma_max^2
        sk      = np.sqrt( np.real( np.vdot( bk.flatten(), b.flatten() ) ) );
        ds      = np.abs( sk - s );
        # Normalize singular vector
        b       = bk / la.norm(bk);
        s       = sk;
    return s;

def LSSolver( y, A, w0, maxItns=1E4, dwAbsTol=1E-4, dfwAbsTol=1E-4 ):
    # Find lipschitz constant
    smax        = OperatorNorm( A );
    L           = 1.01 * math.pow(smax, 2);
    # Initialize variables
    w           = w0;
    dwNrm       = np.inf;
    itn         = 0;
    fw          = 0.5 * math.pow(la.norm((y - A.eval(w, 1)).ravel(), 2), 2);
    # Optimization loop
    while( ( dwNrm > dwAbsTol or np.abs(dfw) > dfwAbsTol ) and itn <= maxItns):
        itn         = itn + 1;
        wk          = w - (1/L) * A.eval(A.eval(w, 1) - y, 2);
        fwk         = 0.5 * math.pow(la.norm((y - A.eval(wk, 1)).ravel(), 2), 2);
        dfw         = fwk - fw;
        dwNrm       = la.norm( (wk - w).ravel(), 2 );
        w           = wk;
        fw          = fwk;
    return w, fw

def pgdl1( t, x0, L, f, df, maxItns=1E4, dwAbsTol=1E-4, dfwAbsTol=1E-5, pdxAbsTol=1E-12, pfxAbsTol=1E-16, disp=0, printEvery=10 ):
    # Project initial iterate on the l1-ball
    w           = project_l1_ball( x0, t, dxAbsTol=pdxAbsTol, fxAbsTol=pfxAbsTol );
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
        wk          = project_l1_ball( v - (1/L) * df( v ), t, dxAbsTol=pdxAbsTol, fxAbsTol=pfxAbsTol );
        sp          = 0.5 * ( 1 + math.pow(1 + 4 * math.pow( s, 2 ), 1/2) );
        vk          = wk + ( ( s - 1 )/sp ) * ( wk - w );
        fwk         = f( wk );
        pwk         = project_l1_ball( wk - (1/L) * df( wk ), t, dxAbsTol=pdxAbsTol, fxAbsTol=pfxAbsTol );
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

def l1reg( t, x0, L, f, df, maxItns=1E4, dwAbsTol=1E-4, dfwAbsTol=1E-5, pdxAbsTol=1E-12, pfxAbsTol=1E-16, disp=0, printEvery=10 ):
    # Project initial iterate on the l1-ball
    w           = x0;
    fw          = f( w ) + t * la.norm( w.ravel(), 1 );
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
            print(' [L1REG] Initial objective: {:+5.3E}'.format(fw) );
        wk          = sft( v - (1/L) * df( v ), 1/t );
        sp          = 0.5 * ( 1 + math.pow(1 + 4 * math.pow( s, 2 ), 1/2) );
        vk          = wk + ( ( s - 1 )/sp ) * ( wk - w );
        fwk         = f( wk ) + t * la.norm( wk.ravel(), 1 );
        dwNrm       = la.norm( (wk - w).ravel(), 2 );
        dfw         = fwk - fw;
        w           = wk;
        v           = vk;
        fw          = fwk;
        s           = sp;
        if( disp == 3 and ( itn == 1 or np.mod( itn, printEvery ) == 0 ) ):
            print(' [L1REG] Iteration: {:03d} | Objective: {:+5.3E} | Last step: {:+5.3E} | Last decrease: {:+5.3E}'.format(itn, fw, dwNrm, dfw) );
    if( disp > 0 ):
        print(' [L1REG Summary] Last iteration: {:03d} | Optimal value: {:+5.3E} | Last step: {:+5.3E} | Last decrease: {:+5.3E}'.format(itn, fw, dwNrm, dfw) );
    return w, f( w )

def CSRecovery( eta, y, A, x0, maxItns=1E3, dfAbsTol=1E-2, dtAbsTol=1E-9, pdxAbsTol=1E-12, pfxAbsTol=1E-16, method='l1reg', disp=0, printEvery=1 ):
    # Parameters
    if( disp == 0 ):
        solver_disp     = 0;
    elif(disp == 1):
        solver_disp     = 0;
    elif(disp == 2):
        solver_disp     = 2;
    else:
        solver_disp     = 3;
    yNrm        = la.norm( y.ravel(), 2 );
    if( yNrm <= eta ):
        return np.zeros( x0.shape );
    # Find lipschitz constant
    smax        = OperatorNorm( A );
    L           = 1.01 * math.pow(smax, 2);
    # Least-squares solution
    wtls, rmax  = LSSolver( y, A, x0 );
    # Lower bound variables
    tmin        = 0
    if( method is 'l1reg' ):
        ftmin       = 0;
        wtmin       = wtls;
    if( method is 'pgdl1' ):
        ftmin       = yNrm;
        wtmin       = np.zeros( x0.shape );
    # Find upper bound variables
    if( method is 'l1reg' ):
        tmax        = la.norm( A.eval(y, 2).ravel(), np.inf );
        ftmax       = 0.5 * math.pow( yNrm, 2 );
        wtmax       = np.zeros( x0.shape );
    if( method is 'pgdl1' ):
        wtmax       = wtls;
        tmax        = la.norm( wtmax.ravel(), 1 );
        ftmax       = 0;
    # Initialize variables
    tsol        = 0.0;
    tend        = 0.0;
    dt          = tmax - tmin;
    dft         = np.abs( ftmin - ftmax );
    itn         = 0;
    while( ( dft > dfAbsTol and dt > dtAbsTol ) and itn <= maxItns ):
        itn         = itn + 1;
        theta       = 0.5;
        tp          = (1-theta) * tmin + theta * tmax;
        if( np.abs(ftmin - eta) < np.abs(eta - ftmax) ):
            wtinit      = wtmin;
        else:
            wtinit      = wtmax;
        # Display
        if( disp > 1 and ( itn == 1 or np.mod(itn, printEvery) == 0 ) ):
            print(' [CS] Iteration {:02d} | eta = {:5.3E}'.format(itn, eta) );
            print('      Interval:      [{:5.3E}, {:5.3E}] ({:5.3E})'.format(tmin, tmax, dt) );
            print('      Trial value:    {:5.3E}             ({:5.3E})'.format(tp, theta) );
            print('      Values:        [{:5.3E}, {:5.3E}] ({:5.3E})'.format(ftmin, ftmax, dft) );
        # Adaptive accuracy
        if( 1E2 <= dft ):
            solver_dwAbsTol     = 1E-2;
            solver_fwAbsTol     = 1E-1;
        elif( 1E-1 <= dft and dft < 1E2 ):
            solver_dwAbsTol     = 1E-3;
            solver_fwAbsTol     = 1E-3;
        else:
            solver_dwAbsTol     = 1E-5;
            solver_fwAbsTol     = 1E-4;
        if( dt < 1 ):
            solver_dwAbsTol     = 1E-5;
            solver_fwAbsTol     = 1E-4;
        if( eta < 1 ):
            solver_dwAbsTol     = np.min([ 1E-5, solver_dwAbsTol ]);
            solver_fwAbsTol     = np.min([ 1E-4, solver_fwAbsTol ]);
        # Solve PGDL1 problem
        tsol_begin  = time.time();
        if( method is 'pgdl1' ):
            wtp, ftp    = pgdl1( tp, wtinit, L, lambda x: f(y, A, x), lambda x: f_grad(y, A, x), dwAbsTol=solver_dwAbsTol, dfwAbsTol=solver_fwAbsTol, pdxAbsTol=pdxAbsTol, pfxAbsTol=pfxAbsTol, disp=solver_disp );
        if( method is 'l1reg' ):
            wtp, ftp    = l1reg( tp, wtinit, L, lambda x: f(y, A, x), lambda x: f_grad(y, A, x), dwAbsTol=solver_dwAbsTol, dfwAbsTol=solver_fwAbsTol, pdxAbsTol=pdxAbsTol, pfxAbsTol=pfxAbsTol, disp=solver_disp );
        tsol_end    = time.time();
        tsol        = tsol_end - tsol_begin;
        tend        = tend + tsol;
        ftp         = math.pow( 2 * ftp, 1/2 );
        # Display
        if( disp == 3 and ( itn == 1 or np.mod(itn, printEvery) == 0 ) ):
            print('      Trial Value:    {:5.3E}'.format(ftp) );
            print('      Elapsed time:   {:8.4f} seconds'.format(tsol) );
        # Update variables
        if( ftp > eta ):
            tmin        = tp;
            ftmin       = ftp;
            wtmin       = wtp;
        else:
            tmax        = tp;
            ftmax       = ftp;
            wtmax       = wtp;
        dt          = tmax - tmin;
        dft         = np.abs( ftmin - ftmax );
    if( disp > 0 ):
        print(' [CS SUMMARY] Last iteration {:02d}'.format(itn) );
        print('      Last interval:      [{:5.3E}, {:5.3E}] ({:5.3E})'.format(tmin, tmax, dt) );
        print('      Optimal value:       {:5.3E}'.format(tp) );
        print('      Residual:            {:5.3E}'.format(ftp) );
        print('      Elapsed time:        {:8.4f} seconds'.format(tend) );
    return wtp, la.norm( wtp.ravel(), 1 )
