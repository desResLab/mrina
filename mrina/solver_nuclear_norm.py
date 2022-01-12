# %% 
import numpy as np
import numpy.linalg as la
import time
import scipy.optimize as sciopt
from mrina.solver_l1_norm import sft, project_l1_ball, MinimizeSumOfSquares

# %%
# Projection onto the nuclear norm ball / tall matrices
def project_nuclear_ball(X, t, eps=1E-5):
    # Compute right singular vectors
    s, V = la.eigh(X.conj().T @ X)
    s = np.sqrt(np.abs(s))

    if np.sum(s) <= t:
        return X

    idx = np.where(s > eps)[0]
    V = V[:, idx]
    s = s[idx]

    st = project_l1_ball(s, t)
    return X @ ((V * (st/s) ) @ V.conj().T)

# Singular value thresholding
def svt(X, t, eps=1E-6):
    # Compute right singular vectors
    s, V = la.eigh(X.conj().T @ X)
    s = np.sqrt(np.abs(s))

    idx = np.where(s >= t)[0]
    if idx.size == 0:
        return np.zeros(shape=X.shape, dtype=X.dtype)

    idx = np.where(s > eps)[0]
    V = V[:, idx]
    s = s[idx]

    st = sft(s, t)
    return X @ ((V * (st/s) ) @ V.conj().T)

# Minimize Sum-of-Squares subject to nuclear norm constraints
#   min_X || A(X) - y ||_2^2 s.t. ||X||_* <= t
def MinimizeSumOfSquaresNuclearBall(t, y, A, Xinit=None, L=None,
                                        maxItns=1E4, 
                                        dxAbsTol=1E-4, dxRelTol=1E-6,
                                        dpAbsTol=1E-5, dpRelTol=1E-8,
                                        disp=False, printEvery=10,
                                        restart=True):
    # Project initial iterate on the nuclear-ball
    if Xinit is None:
        X = np.zeros(A.inShape, dtype=np.complex)
    else:
        X = Xinit
    X = project_nuclear_ball(X, t)
    # Lipschitz constant
    if L is None:
        L = 2.05 * A.norm() ** 2
    # Variables
    rX = A.eval(X) - y
    fX = la.norm(rX.ravel()) ** 2
    gX = 2.0 * A.adjoint(rX)
    rpX = X - project_nuclear_ball(X - gX, t)
    rpXNrm = la.norm(rpX.ravel())
    # Initialize variables
    XNrm = la.norm(X.ravel())
    dXNrm = np.inf
    dfX = np.inf
    itn = 0
    s = 1.0
    Z = X
    # Optimization loop
    stop = False
    if disp:
        print('[PGD-SoS-NuclearBall]')
        print(' Initial objective:   {:1.6e}'.format(fX))
        print(' Initial residual:    {:1.6e}'.format(rpXNrm))
    while not stop:
        itn = itn + 1
        gZ = 2.0 * A.adjoint(A.eval(Z) - y)
        Xp = project_nuclear_ball(Z - gZ/L, t)
        sp = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * s ** 2))
        Zp = Xp + ((s - 1.0)/sp) * (Xp - X)

        # Update
        rXp = A.eval(Xp) - y
        fXp = la.norm(rXp.ravel()) ** 2

        dfX = fXp - fX
        dXNrm = la.norm((Xp - X).ravel())

        if restart and dfX > 1E-12:
            Z = X
            s = 1.0
            dfX = np.inf
            dXNrm = np.inf
        else:
            X = Xp
            gX = 2.0 * A.adjoint(rXp)
            rpX = X - project_nuclear_ball(X - gX, t)
            rpXNrm = la.norm(rpX.ravel())

            Z = Zp
            fX = fXp
            s = sp

        XNrm = la.norm(X.ravel())
        dXTol = np.max([ dxAbsTol, dxRelTol * XNrm ])
        rpXTol = np.max([ dpAbsTol, dpRelTol * XNrm ]) 
        if (dXNrm < dXTol and rpXNrm < rpXTol) or itn > maxItns:
            stop = True
        if disp and (itn == 1 or np.mod(itn, printEvery) == 0 or stop):
            print(' {:04d} | fX: {:1.3E} | dfX: {:+1.3E} | dXNrm: {:1.3E} | rpXNrm: {:5.3E}'.format(itn, fX, dfX, dXNrm, rpXNrm))
    return X, fX

# Minimize nuclear norm regularized Sum-of-Squares
#   min_X || A(X) - y ||_2^2 + t ||X||_*
def MinimizeNNDN(t, y, A, Xinit=None, L=None,
                    maxItns=1E4, 
                    dxAbsTol=1E-4, dxRelTol=1E-6,
                    dpAbsTol=1E-5, dpRelTol=1E-8,
                    disp=False, printEvery=10,
                    restart=True):
    # Project initial iterate on the nuclear-ball
    if Xinit is None:
        X = np.zeros(A.inShape, dtype=np.complex)
    else:
        X = Xinit
    # Lipschitz constant
    if L is None:
        L = 2.05 * A.norm() ** 2
    # Variables
    rX = A.eval(X) - y
    fX = la.norm(rX.ravel()) ** 2 + t * la.norm(np.sqrt(la.eigvalsh(X.conj().T @ X)), 1)
    gX = 2.0 * A.adjoint(rX)
    rpX = X - svt(X - gX, t)
    rpXNrm = la.norm(rpX.ravel())
    # Initialize variables
    XNrm = la.norm(X.ravel())
    dXNrm = np.inf
    dfX = np.inf
    itn = 0
    s = 1.0
    Z = X
    # Optimization loop
    stop = False
    if disp:
        print('[PGD-NNDN]')
        print(' Initial objective:   {:1.6e}'.format(fX))
        print(' Initial residual:    {:1.6e}'.format(rpXNrm))
    while not stop:
        itn = itn + 1
        gZ = 2.0 * A.adjoint(A.eval(Z) - y)
        Xp = svt(Z - gZ/L, t/L)
        sp = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * s ** 2))
        Zp = Xp + ((s - 1.0)/sp) * (Xp - X)

        # Update
        rXp = A.eval(Xp) - y
        fXp = la.norm(rXp.ravel()) ** 2 + t * la.norm(np.sqrt(la.eigvalsh(Xp.conj().T @ Xp)), 1)

        dfX = fXp - fX
        dXNrm = la.norm((Xp - X).ravel())
        if restart and dfX > 1E-12:
            Z = X
            s = 1.0
            dfX = -np.inf
            dXNrm = np.inf
        else:
            X = Xp
            gX = 2.0 * A.adjoint(rXp)
            rpX = X - svt(X - gX, t)
            rpXNrm = la.norm(rpX.ravel())

            Z = Zp
            fX = fXp
            s = sp

        dXTol = np.max([ dxAbsTol, dxRelTol * XNrm ])
        rpXTol = np.max([ dpAbsTol, dpRelTol * XNrm ]) 
        if (dXNrm < dXTol and rpXNrm < rpXTol) or itn > maxItns:
            stop = True

        if disp and (itn == 1 or np.mod(itn, printEvery) == 0 or stop):
            print(' {:04d} | obj: {:1.3E} | dfx: {:+1.3E} | dxNrm: {:1.3E} | rpxNrm: {:5.3E}'.format(itn, fX, dfX, dXNrm, rpXNrm))
    return X, fX

class RootSolverNuclearNormNoisy():
    def __init__(self, eta, y, A,
                    maxItns=1E4, 
                    dxAbsTol=1E-4, dxRelTol=1E-6,
                    dpAbsTol=1E-5, dpRelTol=1E-8,
                    disp=False, printEvery=10,
                    restart=True,
                    method='NNDN'):
        # Problem parameters
        self.eta = eta
        self.y = y
        self.A = A

        self.ANrm = A.norm()
        self.L = 2.05 * self.ANrm ** 2

        # For bracketing
        self._itn = 0
        self._t = [ None, None ]
        self._ft = [ None, None ]
        self._Xinit = [ None, None ]
        if method == 'SoS-NucBall':
            # Lower bound
            self._t[0] = 0.0
            self._ft[0] = la.norm(y.ravel()) - eta
            self._Xinit[0] = np.zeros(A.inShape, dtype=np.complex)
            # Upper bound
            _Xopt, _fopt = MinimizeSumOfSquares(y, A, 
                                                maxItns=maxItns, 
                                                dxAbsTol=dxAbsTol, dxRelTol=dxRelTol,
                                                gradAbsTol=dpAbsTol, gradRelTol=dpRelTol,
                                                disp=False)
            self._t[1] = la.norm(np.sqrt(la.eigvalsh(_Xopt.conj().T @ _Xopt)), 1)
            self._ft[1] = la.norm((A.eval(_Xopt) - y).ravel(), 2) - self.eta
            self._Xinit[1] = _Xopt
        elif method == 'NNDN':
            # Lower bound
            _Xopt, _fopt = MinimizeSumOfSquares(y, A, 
                                                maxItns=maxItns, 
                                                dxAbsTol=dxAbsTol, dxRelTol=dxRelTol,
                                                gradAbsTol=dpAbsTol, gradRelTol=dpRelTol,
                                                disp=False)            
            self._t[0] = 0.0
            self._ft[0] = -eta
            self._Xinit[0] = _Xopt
            # Upper bound
            _Z = A.adjoint(y)
            self._t[1] = la.norm(np.sqrt(la.eigvalsh(_Z.conj().T @ _Z)), np.inf)
            self._ft[1] = la.norm(y.ravel(), 2) - self.eta
            self._Xinit[1] = np.zeros(A.inShape, dtype=np.complex)
        else:
            raise ValueError('Method can be either SoS-NucBall or NNDN.')
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
        Xinit = theta * self._Xinit[0] + (1.0 - theta) * self._Xinit[1]

        if self.method == 'SoS-NucBall':
            Xopt, _ = MinimizeSumOfSquaresNuclearBall(t, y, A, Xinit=Xinit, L=self.L,
                                                        maxItns=self.maxItns, 
                                                        dxAbsTol=self.dxAbsTol, dxRelTol=self.dxRelTol,
                                                        dpAbsTol=self.dpAbsTol, dpRelTol=self.dpRelTol,
                                                        disp=False, printEvery=self.printEvery,
                                                        restart=self.restart)
        elif self.method == 'NNDN':
            Xopt, _ = MinimizeNNDN(t, y, A, Xinit=Xinit, L=self.L,
                                    maxItns=self.maxItns, 
                                    dxAbsTol=self.dxAbsTol, dxRelTol=self.dxRelTol,
                                    dpAbsTol=self.dpAbsTol, dpRelTol=self.dpRelTol,
                                    disp=False, printEvery=self.printEvery,
                                    restart=self.restart)
        
        if t > self._t[1]:
            self._t[0] = self._t[1]
            self._ft[0] = self._ft[1]
            self._Xinit[0] = self._Xinit[1]

            self._t[1] = t
            self._ft[1] = la.norm((A.eval(Xopt) - y).ravel(), 2) - self.eta
            self._Xinit[1] = Xopt
        elif t < self._t[0]:
            self._t[1] = self._t[0]
            self._ft[1] = self._ft[0]
            self._Xinit[1] = self._Xinit[0]

            self._t[0] = t
            self._ft[0] = la.norm((A.eval(Xopt) - y).ravel(), 2) - self.eta
            self._Xinit[0] = Xopt
        else:
            self._t[tNear] = t
            self._ft[tNear] = la.norm((A.eval(Xopt) - y).ravel(), 2) - self.eta
            self._Xinit[tNear] = Xopt

        return la.norm((A.eval(Xopt) - y).ravel(), 2) - self.eta

def RecoveryNuclearNormNoisy(eta, y, A, 
                    maxItns=1E4, 
                    dxAbsTol=1E-4, dxRelTol=1E-6,
                    dpAbsTol=1E-5, dpRelTol=1E-8,
                    disp=False, printEvery=100,
                    method='NNDN'):

    # Create solver object
    T = RootSolverNuclearNormNoisy(eta, y, A,
                                    maxItns=maxItns, 
                                    dxAbsTol=dxAbsTol, dxRelTol=dxRelTol,
                                    dpAbsTol=dpAbsTol, dpRelTol=dpRelTol,
                                    disp=disp, printEvery=printEvery,
                                    restart=True, method=method)
    if disp:
        print('[RECOVERY: NUCLEAR NORM-NOISY]' )
        print(' Method:          {:s}'.format(method) )
        print(' Bound:           {:1.4E}'.format(eta) )
        print(' Initial values:')
        print('     t           [{:1.4E},   {:1.4E}]'.format(T._t[0], T._t[1]))
        print('     f(t)        [{:+1.4E}, {:+1.4E}]'.format(T._ft[0], T._ft[1]))
    # Root finding via TOMS748
    tStart = time.time()
    tMin = T._t[0]
    tMax = T._t[1]
    topt, rnfo = sciopt.toms748(lambda t: T.solve(t), tMin, tMax, xtol=5E-4, full_output=disp, disp=disp)
    if np.abs(topt - T._t[0]) < np.abs(topt - T._t[1]):
        Xopt = T._Xinit[0]
    else:
        Xopt = T._Xinit[1]
    tStop = time.time() - tStart
    if disp:
        print('  Summary');
        print('      Iterations:          {:d}'.format(rnfo.iterations) )
        print('      Function calls:      {:d}'.format(rnfo.function_calls) )
        print('      Optimal value:       {:5.3E}'.format(topt) )
        print('      Elapsed time:        {:8.4f} seconds'.format(tStop) )
    return Xopt, la.norm(Xopt.ravel(), 1)

# %%
if __name__ == '__main__':
    # Import
    from maps import OperatorFourierLowRank
    
    print('[TESTING SOLVER ROUTINES]')
    # *** EXPERIMENTS FOR SINGLE IMAGE AND SUBSAMPLING - L1-NORM
    # Set up map
    imShape = (128, 128)
    samplingSet = np.where(np.random.uniform(size=imShape) < 0.75, True, False)
    A = OperatorFourierLowRank(imShape, samplingSet=samplingSet)
    xinit = np.random.normal(size=A.inShape) + 1j * np.random.normal(size=A.inShape)
    y = A.eval(xinit)
    print('[SINGLE IMAGE - RANDOM UNDERSAMPLING - L1-NORM RECOVERY]')
    print('    Input shape:    {:d} x {:d}'.format(A.inShape[0], A.inShape[1]))
    print('    Output shape:   {:d} x {:d}'.format(A.outShape[0], 1))
    print('    Operator norm:  {:1.3e}'.format(A.norm()))
    # Tests
    print('\n[TEST: MinimizeSumOfSquares]')
    xsol, fmin = MinimizeSumOfSquares(y, A, disp=True)
    print('\n[TEST: MinimizeSumOfSquaresNuclearBall]')
    print('  ---> with no restart...')
    xsol, fmin = MinimizeSumOfSquaresNuclearBall(1.0, y, A, disp=True, restart=False, printEvery=10)
    print('  ---> with restart...')
    _xsol, _fmin = MinimizeSumOfSquaresNuclearBall(1.0, y, A, disp=True, restart=True)
    print('  ---> Difference:     {:1.5e}'.format(la.norm((xsol - _xsol).ravel())))
    print('\n[TEST: RecoveryL1Noisy - SoS-NucBall]')
    xsol, fmin = RecoveryNuclearNormNoisy(2.0, y, A, maxItns=1E4, disp=True, printEvery=100, method='SoS-NucBall')
    print('\n[TEST: RecoveryL1Noisy - NNDN]')
    _xsol, _fmin = RecoveryNuclearNormNoisy(2.0, y, A, maxItns=1E4, disp=True, printEvery=100, method='NNDN')
    print('  ---> Difference:')
    print('         Objective:    {:1.5e}'.format(np.abs(fmin - _fmin)))
    print('         Solution:     {:1.5e}'.format(la.norm((xsol - _xsol).ravel())))

# %%
