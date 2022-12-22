# %% 
import numpy as np
import numpy.linalg as la
import time
import scipy.optimize as sciopt

# %%
# Soft-thresholding
def sft(x, t):
    return x * (1.0 - np.minimum(t / np.maximum(np.abs(x), 1E-32), 1.0))

# Projection onto the l1-ball
def project_l1_ball(x, t):
    xL1Nrm  = la.norm(x.ravel(), 1)
    if xL1Nrm > t:
        # If the input is outside the l1-ball we project.
        nx = x.size;
        # Sort magnitudes in decreasing order
        mx = np.flip(np.sort(np.absolute(x.ravel()), kind='mergesort'))
        # Obtain unique values (this is needed for some borderline cases)
        umx = np.flip(np.unique(mx))
        if t <= umx[0] - umx[1]:
            # If the radius of the l1-ball is small and the input is far away
            # and the shrinkage is large.
            vp = umx[0]
            sroot = vp - t
            return sft(x, sroot)
        else:
            # This is the average case
            cmx = np.cumsum(mx)
            smx = cmx - np.array(range(1, nx+1)) * mx
        if smx[nx - 1] < t:
            # This condition handles some cases when the input is close to
            # the boundary and the shrinkage is small.
            sroot = (xL1Nrm - t) / nx
            return sft(x, sroot)
        idp, = np.where( smx > t )
        idp = idp[0];
        sroot = (cmx[idp - 1] - t) / idp
        return sft(x, sroot)
    # If the input is within the l1-ball we do nothing.
    return x

# Minimize Sum-of-Squares
#   min_x || Ax - y ||_2^2
def MinimizeSumOfSquares(y, A, xinit=None, 
                            maxItns=1E4, 
                            dxAbsTol=1E-6, dxRelTol=1E-9,
                            gradAbsTol=1E-5, gradRelTol=1E-8,
                            disp=False, printEvery=100):
    # Find Lipschitz constant of the gradient
    L = 2.05 * A.norm() ** 2
    # Initialize variables
    if xinit is None:
        x = np.zeros(A.wavShape, dtype=complex)
    else:
        x = xinit
    xNrm = la.norm(x.ravel(), 2)
    dxNrm = np.inf
    itn = 0
    fx = la.norm((y - A.eval(x)).ravel(), 2) ** 2
    # Optimization loop
    stop = False
    if disp:
        print('[MinimizeSumOfSquares]')
        print(' Initial objective:   {:1.6e}'.format(fx))
    while not stop: 
        itn = itn + 1
        xp = x - (2.0/L) * A.adjoint(A.eval(x) - y)
        fxp = la.norm((y - A.eval(xp)).ravel(), 2) ** 2
        dxNrm = la.norm((xp - x).ravel(), 2)
        gradNrm = L * dxNrm
        dxNrmTol = np.max([ dxAbsTol, 
                            dxRelTol * xNrm, 
                            L * gradAbsTol, 
                            L * gradRelTol * xNrm ])
        if dxNrm < dxNrmTol or itn > maxItns:
            stop = True
        x = xp
        fx = fxp
        xNrm = la.norm(x.ravel(), 2)
        if disp and (itn == 1 or np.mod(itn, printEvery) == 0 or stop):
            print(' {:04d} | obj: {:1.3E} | gradNrm: {:1.3E}'.format(itn, fx, gradNrm))
    return x, fx

# Minimize Sum-of-Squares subject to l1-norm
#   min_x || Ax - y ||_2^2 s.t. ||x||_1 <= t
def MinimizeSumOfSquaresL1Ball(t, y, A, xinit=None, L=None,
                                maxItns=1E4, 
                                dxAbsTol=1E-4, dxRelTol=1E-6,
                                dpAbsTol=1E-5, dpRelTol=1E-8,
                                disp=False, printEvery=10,
                                restart=True):
    # Project initial iterate on the l1-ball
    if xinit is None:
        x = np.zeros(A.inShape, dtype=complex)
    else:
        x = project_l1_ball(xinit, t)
    if L is None:
        L = 2.05 * A.norm() ** 2
    rx = A.eval(x) - y
    fx = la.norm(rx.ravel()) ** 2
    gx = 2.0 * A.adjoint(rx)
    rpx = x - project_l1_ball(x - gx/L, t)
    rpxNrm = la.norm(rpx.ravel())
    # Initialize variables
    xNrm = la.norm(x.ravel())
    dxNrm = np.inf
    dfx = -np.inf
    itn = 0
    s = 1
    z = x
    # Optimization loop
    stop = False
    if disp:
        print('[PGD-SoS-L1Ball]')
        print(' Initial objective:   {:1.6e}'.format(fx))
        print(' Initial residual:    {:1.6e}'.format(rpxNrm))
    while not stop:
        itn = itn + 1
        gz = 2.0 * A.adjoint(A.eval(z, 1) - y)
        xp = project_l1_ball(z - gz/L, t)
        sp = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * s ** 2))
        zp = xp + ((s - 1.0)/sp) * (xp - x)

        # Update
        rxp = A.eval(xp) - y
        fxp = la.norm(rxp.ravel()) ** 2

        dfx = fxp - fx
        dxNrm = la.norm((xp - x).ravel())

        if restart and dfx > 1E-12:
            z = x
            s = 1.0
            dfx = np.inf
            dxNrm = np.inf
        else:
            x = xp
            gx = 2.0 * A.adjoint(rxp)
            rpx = x - project_l1_ball(x - gx/L, t)
            rpxNrm = la.norm(rpx.ravel())

            z = zp
            fx = fxp
            s = sp

        xNrm = la.norm(x.ravel())
        dxTol = np.max([ dxAbsTol, dxRelTol * xNrm ])
        rpxTol = np.max([ dpAbsTol, dpRelTol * xNrm ]) 
        if (dxNrm < dxTol and rpxNrm < rpxTol) or itn > maxItns:
            stop = True
        if disp and (itn == 1 or np.mod(itn, printEvery) == 0 or stop):
            print(' {:04d} | obj: {:1.3E} | dfx: {:+1.3E} | dxNrm: {:1.3E} | rpxNrm: {:5.3E}'.format(itn, fx, dfx, dxNrm, rpxNrm))
    return x, fx

# Minimize Sum-of-Squares
#   min_x || Ax - y ||_2^2 + t ||x||_1
def MinimizeBPDN(t, y, A, xinit=None, L=None,
                    maxItns=1E4, 
                    dxAbsTol=1E-4, dxRelTol=1E-6,
                    dpAbsTol=1E-5, dpRelTol=1E-8,
                    disp=False, printEvery=10,
                    restart=True):
    # Project initial iterate on the l1-ball
    if xinit is None:
        x = np.zeros(A.inShape, dtype=complex)
    else:
        x = project_l1_ball(xinit, t)
    z = x
    if L is None:
        L = 2.05 * A.norm() ** 2
    rx = A.eval(x) - y
    fx = la.norm(rx.ravel()) ** 2 + t * la.norm(x.ravel(), 1)
    gx = 2.0 * A.adjoint(rx)
    rpx = x - sft(x - gx, t)
    rpxNrm = la.norm(rpx.ravel())
    # Initialize variables
    xNrm = la.norm(x.ravel())
    dxNrm = np.inf
    dfx = -np.inf
    itn = 0
    s = 1.0
    # Optimization loop
    stop = False
    if disp:
        print('[PGD-BPDN]')
        print(' Initial objective:   {:1.6e}'.format(fx))
        print(' Initial residual:    {:1.6e}'.format(rpxNrm))
    while not stop:
        itn = itn + 1
        gz = 2.0 * A.adjoint(A.eval(z, 1) - y)
        xp = sft(z - gz/L, t/L)
        sp = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * s ** 2))
        zp = xp + ((s - 1.0)/sp) * (xp - x)

        # Update
        rxp = A.eval(xp) - y
        fxp = la.norm(rxp.ravel()) ** 2 + t * la.norm(xp.ravel(), 1)

        dfx = fxp - fx
        dxNrm = la.norm((xp - x).ravel())

        if restart and dfx > 1E-12:
            z = x
            s = 1.0
            dfx = np.inf
            dxNrm = np.inf
        else:
            x = xp
            gx = 2.0 * A.adjoint(rxp)
            rpx = x - sft(x - gx, t)
            rpxNrm = la.norm(rpx.ravel())

            z = zp
            fx = fxp
            s = sp

        xNrm = la.norm(x.ravel())
        dxTol = np.max([ dxAbsTol, dxRelTol * xNrm ])
        rpxTol = np.max([ dpAbsTol, dpRelTol * xNrm ]) 
        if (dxNrm < dxTol and rpxNrm < rpxTol) or itn > maxItns:
            stop = True

        if disp and (itn == 1 or np.mod(itn, printEvery) == 0 or stop):
            print(' {:04d} | obj: {:1.3E} | dfx: {:+1.3E} | dxNrm: {:1.3E} | rpxNrm: {:5.3E}'.format(itn, fx, dfx, dxNrm, rpxNrm))
    return x, fx

class RootSolverL1NormNoisy():
    def __init__(self, eta, y, A,
                    maxItns=1E4, 
                    dxAbsTol=1E-4, dxRelTol=1E-6,
                    dpAbsTol=1E-5, dpRelTol=1E-8,
                    disp=False, printEvery=10,
                    restart=True,
                    method='SoS-L1Ball', disp_method=False):
        # Problem parameters
        self.eta = eta
        self.y = y
        self.A = A
        self.disp_method = disp_method

        self.ANrm = A.norm()
        self.L = 2.05 * self.ANrm ** 2

        # For bracketing
        self._itn = 0
        self._t = [ None, None ]
        self._ft = [ None, None ]
        self._xinit = [ None, None ]

        _xopt, _ = MinimizeSumOfSquares(y, A, 
                                            maxItns=maxItns, 
                                            dxAbsTol=dxAbsTol, dxRelTol=dxRelTol,
                                            gradAbsTol=dpAbsTol, gradRelTol=dpRelTol,
                                            disp=False)
        _ropt = A.eval(_xopt) - y
        if method == 'SoS-L1Ball':
            # Lower bound
            self._t[0] = 0.0
            self._ft[0] = la.norm(y.ravel()) - eta
            self._xinit[0] = np.zeros(A.inShape, dtype=complex)
            # Upper bound
            self._t[1] = la.norm(_xopt.ravel(), 1)
            self._ft[1] = la.norm(_ropt.ravel(), 2) - eta
            self._xinit[1] = _xopt
        elif method == 'BPDN':
            # Lower bound
            self._t[0] = 0.0
            self._ft[0] = -eta
            self._xinit[0] = _xopt
            # Upper bound
            _z = A.adjoint(y)
            self._t[1] = la.norm(_z.ravel(), np.inf)
            self._ft[1] = la.norm(y.ravel(), 2) - eta
            self._xinit[1] = np.zeros(A.inShape, dtype=complex)
        else:
            raise ValueError('Method can be either SoS-L1Ball or BPDN.')
        self.method = method
        # Solver options
        self.restart = restart
        # Solver parameters
        self.maxItns = maxItns
        self.dxAbsTol = dxAbsTol
        self.dxRelTol = dxRelTol
        self.dpAbsTol = dpAbsTol
        self.dpRelTol = dpRelTol
        # Display options
        self.disp = disp
        self.printEvery = printEvery

    def solve(self, t):
        self._itn = self._itn + 1
        if self.disp:
            print(' {:02d} | t* = {:1.4E} in [{:1.4E}, {:1.4E}] | f: [{:+1.4E}, {:+1.4E}]'.format(self._itn, t, self._t[0], self._t[1], self._ft[0], self._ft[1]))

        if np.abs(t - self._t[0]) < np.abs(t - self._t[1]):
            tNear = 0
        else:
            tNear = 1

        theta = np.abs(t - self._t[0]) / (np.abs(t - self._t[0]) + np.abs(t - self._t[1]))
        xinit = theta * self._xinit[0] + (1.0 - theta) * self._xinit[1]

        if self.method == 'SoS-L1Ball':
            xopt, _ = MinimizeSumOfSquaresL1Ball(t, self.y, self.A, xinit=xinit, L=self.L,
                                                    maxItns=self.maxItns, 
                                                    dxAbsTol=self.dxAbsTol, dxRelTol=self.dxRelTol,
                                                    dpAbsTol=self.dpAbsTol, dpRelTol=self.dpRelTol,
                                                    disp=self.disp_method, printEvery=self.printEvery,
                                                    restart=self.restart)
        elif self.method == 'BPDN':
            xopt, _ = MinimizeBPDN(t, self.y, self.A, xinit=xinit, L=self.L,
                                    maxItns=self.maxItns, 
                                    dxAbsTol=self.dxAbsTol, dxRelTol=self.dxRelTol,
                                    dpAbsTol=self.dpAbsTol, dpRelTol=self.dpRelTol,
                                    disp=self.disp_method, printEvery=self.printEvery,
                                    restart=self.restart)

        ft = la.norm((self.A.eval(xopt) - self.y).ravel(), 2) - self.eta

        if t > self._t[1]:
            self._t[0] = self._t[1]
            self._ft[0] = self._ft[1]
            self._xinit[0] = self._xinit[1]

            self._t[1] = t
            self._ft[1] = ft
            self._xinit[1] = xopt
        elif t < self._t[0]:
            self._t[1] = self._t[0]
            self._ft[1] = self._ft[0]
            self._xinit[1] = self._xinit[0]

            self._t[0] = t
            self._ft[0] = ft
            self._xinit[0] = xopt
        else:
            self._t[tNear] = t
            self._ft[tNear] = ft
            self._xinit[tNear] = xopt

        return ft

def RecoveryL1NormNoisy(eta, y, A, 
                    maxItns=1E4, 
                    dxAbsTol=1E-4, dxRelTol=1E-6,
                    dpAbsTol=1E-5, dpRelTol=1E-8,
                    disp=False, printEvery=100,
                    method='BPDN', disp_method=False):

    # Create solver object
    T = RootSolverL1NormNoisy(eta, y, A,
                            maxItns=maxItns, 
                            dxAbsTol=dxAbsTol, dxRelTol=dxRelTol,
                            dpAbsTol=dpAbsTol, dpRelTol=dpRelTol,
                            disp=disp, printEvery=printEvery,
                            restart=True, method=method, disp_method=disp_method)
    if disp:
        print('[RECOVERY: L1-NOISY]' )
        print(' Method:          {:s}'.format(method) )
        print(' Bound:           {:1.4E}'.format(eta) )
        print(' Initial values:')
        print('     t           [{:1.4E},   {:1.4E}]'.format(T._t[0], T._t[1]))
        print('     f(t)        [{:+1.4E}, {:+1.4E}]'.format(T._ft[0], T._ft[1]))
    # Root finding via TOMS748
    tStart = time.time()
    tMin = T._t[0]
    tMax = T._t[1]
    topt, rnfo = sciopt.toms748(lambda t: T.solve(t), tMin, tMax, k=1, xtol=1E-4, full_output=disp, disp=disp)
    if np.abs(topt - T._t[0]) < np.abs(topt - T._t[1]):
        xopt = T._xinit[0]
    else:
        xopt = T._xinit[1]
    tStop = time.time() - tStart
    if disp:
        print('  Summary');
        print('      Iterations:          {:d}'.format(rnfo.iterations) )
        print('      Function calls:      {:d}'.format(rnfo.function_calls) )
        print('      Optimal value:       {:5.3E}'.format(topt) )
        print('      Elapsed time:        {:8.4f} seconds'.format(tStop) )
    return xopt, la.norm(xopt.ravel(), 1)

# %%
if __name__ == '__main__':
    # Import
    from maps import OperatorWaveletToFourier, OperatorWaveletToFourierX4
    
    do_single_image = True
    do_x4_images = False

    print('[TESTING SOLVER ROUTINES]')
    # Set up map
    imShape = (128, 128)
    # *** EXPERIMENTS FOR SINGLE IMAGE AND SUBSAMPLING - L1-NORM
    if do_single_image:
        samplingSet = np.where(np.random.uniform(size=imShape) < 0.75, True, False)
        A = OperatorWaveletToFourier(imShape, samplingSet=samplingSet, waveletName='db4')
        xinit = np.random.normal(size=A.inShape) + 1j * np.random.normal(size=A.inShape)
        y = A.eval(xinit)
        print('[SINGLE IMAGE - RANDOM UNDERSAMPLING - L1-NORM RECOVERY]')
        print('    Input shape:    {:d} x {:d}'.format(A.inShape[0], A.inShape[1]))
        print('    Output shape:   {:d} x {:d}'.format(A.outShape[0], 1))
        print('    Operator norm:  {:1.3e}'.format(A.norm()))
        # Tests
        print('\n[TEST: MinimizeSumOfSquares]')
        xsol, fmin = MinimizeSumOfSquares(y, A, disp=True)

        print('\n[TEST: MinimizeSumOfSquaresL1Ball]')
        print('  ---> with no restart...')
        xsol, fmin = MinimizeSumOfSquaresL1Ball(0.25, y, A, disp=True, restart=False)
        print('  ---> with restart...')
        _xsol, _fmin = MinimizeSumOfSquaresL1Ball(0.25, y, A, disp=True, restart=True)
        print('  ---> Difference:     {:1.5e}'.format(la.norm((xsol - _xsol).ravel())))

        print('\n[TEST: RecoveryL1Noisy - SoS-L1Ball]')
        xsol, fmin = RecoveryL1NormNoisy(10, y, A, maxItns=1E4, disp=True, printEvery=100, method='SoS-L1Ball')

        print('\n[TEST: RecoveryL1Noisy - BPDN]')
        _xsol, _fmin = RecoveryL1NormNoisy(10, y, A, maxItns=1E4, disp=True, printEvery=100, method='BPDN')
        print('  ---> Difference:')
        print('         Objective:    {:1.5e}'.format(np.abs(fmin - _fmin)))
        print('         Solution:     {:1.5e}'.format(la.norm((xsol - _xsol).ravel())))

    # *** EXPERIMENTS FOR MULTIPLE IMAGES AND SUBSAMPLING - L1-NORM
    if do_x4_images:
        samplingSet = np.where(np.random.uniform(size=imShape + (4,)) < 0.75, True, False)
        A = OperatorWaveletToFourierX4(imShape, samplingSet=samplingSet, waveletName='db4')
        xinit = np.random.normal(size=A.inShape) + 1j * np.random.normal(size=A.inShape)
        y = A.eval(xinit)
        print('[4 IMAGES - RANDOM UNDERSAMPLING - L1-NORM RECOVERY]')
        print('    Input shape:    {:d} x {:d} x {:d}'.format(A.inShape[0], A.inShape[1], A.inShape[2]))
        print('    Output shape:   {:d} x {:d}'.format(A.outShape[0], 1))
        print('    Operator norm:  {:1.3e}'.format(A.norm()))
        # Tests
        print('\n[TEST: MinimizeSumOfSquares]')
        xsol, fmin = MinimizeSumOfSquares(y, A, disp=True)

        print('\n[TEST: MinimizeSumOfSquaresL1Ball]')
        print('  ---> with no restart...')
        xsol, fmin = MinimizeSumOfSquaresL1Ball(0.25, y, A, disp=True, restart=False)
        print('  ---> with restart...')
        _xsol, _fmin = MinimizeSumOfSquaresL1Ball(0.25, y, A, disp=True, restart=True)
        print('  ---> Difference:     {:1.5e}'.format(la.norm((xsol - _xsol).ravel())))

        print('\n[TEST: RecoveryL1Noisy - SoS-L1Ball]')
        xsol, fmin = RecoveryL1NormNoisy(10, y, A, maxItns=1E4, disp=True, printEvery=100, method='SoS-L1Ball')

        print('\n[TEST: RecoveryL1Noisy - BPDN]')
        _xsol, _fmin = RecoveryL1NormNoisy(10, y, A, maxItns=1E4, disp=True, printEvery=100, method='BPDN')
        print('  ---> Difference:')
        print('         Objective:    {:1.5e}'.format(np.abs(fmin - _fmin)))
        print('         Solution:     {:1.5e}'.format(la.norm((xsol - _xsol).ravel())))
# %%
