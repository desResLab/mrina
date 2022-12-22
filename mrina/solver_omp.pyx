import sys
import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport sqrt

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
__docformat__ = 'restructuredtext'

# Simple shortcuts---linalg.norm is too slow for small vectors
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef normof2(double x,double y): 
  return sqrt(x*x + y*y)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef normof4(double x1, double x2, double x3, double x4): 
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
        self.x = None
        self.var = None
        self.itn = 0
        self.istop = 0
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

    # @cython.boundscheck(False)
    # @cython.wraparound(False)
    # @cython.nonecheck(False)
    def solve(self, b, long itnlim=0, double damp=0.0, M=None, N=None, double atol=1.0e-9,
                double btol=1.0e-9, double conlim=1.0e+8, show=False, wantvar=False, **kwargs):
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
        cdef int m
        cdef int n
        m, n = A.shape        

        # print('---')
        # print('Operator Size in lsQR Solve: m=%5d n=%5d' % (m,n))
        # print('---')

        if itnlim == 0: itnlim = 3*n

        if wantvar:
            var = np.zeros(n,1,dtype=complex)
        else:
            var = None

        cdef double dampsq = damp*damp

        cdef int itn = 0
        cdef int istop = 0
        cdef double ctol = 0.0
        if conlim > 0.0: self.ctol = 1.0/conlim
        cdef double Anorm = 0.0
        cdef double Acond = 0.0
        cdef double z = 0.0
        cdef double xnorm = 0.0
        cdef double xxnorm = 0.0
        cdef double ddnorm = 0.0
        cdef double res2 = 0.0
        cdef double cs2 = -1.0
        cdef double sn2 = 0.0

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

        x = np.zeros(n,dtype=complex)
        cdef double xNrgNorm2 = 0.0   # Squared energy norm of final solution.
        dErr = np.zeros(window)       # Truncated direct error terms.
        cdef double trncDirErr = 0.0  # Truncated direct error.

        if store_iterates:
            self.iterates.append(x.copy())

        Mu = b[:m].copy()
        if M is not None:
            u = M(Mu)
        else:
            u = Mu

        cdef double alpha = 0.0
        # beta = np.sqrt(np.dot(u,Mu))       # norm(u)
        cdef double beta = np.linalg.norm(u)
        if beta > 0:
            u /= beta
            if M is not None: Mu /= beta
            ###
            tmp = np.zeros(A.imShape,dtype=complex)
            tmp[A.samplingSet] = u[:]
            Nv = A.adjoint(tmp)[A.basisSet]
            ###
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

        cdef double rhobar = alpha
        cdef double phibar = beta
        cdef double bnorm  = beta
        cdef double rnorm  = beta

        cdef double r1norm = rnorm
        cdef double r2norm = rnorm
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
            
            ###
            tmp = np.zeros(A.wavShape, dtype=complex)
            tmp[A.basisSet] = v[:]
            Mu = A.eval(tmp)[A.samplingSet] - alpha*Mu
            ###
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
                ###
                tmp = np.zeros(A.imShape,dtype=complex)
                tmp[A.samplingSet] = u[:]
                Nv = A.adjoint(tmp)[A.basisSet] - beta*Nv
                ###
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
                test2 = np.inf
            else:
                test2 = Arnorm/(Anorm * rnorm)

            if Acond == 0.0:
                test3 = np.inf
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

def OMPRecovery(A, b, double tol=1E-6, showProgress=True, int progressInt=1, maxItns=10, 
                ompMethod='stomp',ts_factor=2.0, useNorms=False):

  #gives the OMP solution x given the matrix A and vector b and error 
  #parameter, Tr, to find a stopping point
  cdef long m = 0
  cdef long n = 0
  m,n = A.shape

  # Set the maximum number of iterations as per omp
  if maxItns is None:
    maxItns=m

  # Create a new vector for the residual starting from b
  vec_b = b[A.samplingSet]
  curr_res = vec_b.copy()
  cdef double iniResNorm = 0.0
  iniResNorm = np.linalg.norm(curr_res, ord=2)

  # Intialize Empty index set
  # indexSet = [loopA for loopA in range(2500,4096,1)]
  indexSet = []
  notIndexSet = [loopA for loopA in range(n)]

  # Initialize Solution to Zero
  ompSol = np.zeros(n,dtype=complex) 

  # Determine Column Norms
  if(useNorms):
  # Anorm = np.ones(np.prod(A.wavShape))
  # if(False):
    Anorm = np.zeros(np.prod(A.wavShape))
    print('Pre-computing Operator Column Norms...')
    for loopA in range(A.wavShape[0]):
      print('%d/%d' % (loopA,A.wavShape[0]))
      for loopB in range(A.wavShape[1]):
        unit = np.zeros(A.wavShape)
        unit[loopA,loopB] = 1.0
        Anorm[loopA * A.wavShape[0] + loopB] = np.linalg.norm(A.eval(unit))**2
        # print('%15f' % (Anorm[loopA * A.wavShape[0] + loopB]),end='')
    print('OK')
  
  # Print Header
  if(showProgress):
    if(ompMethod == 'omp'):
      print('%10s %15s' % ('OMP Iter.','Res. Norm'))
    elif(ompMethod == 'stomp'):
      print('%10s %15s %15s %8s' % ('OMP Iter.','Res. Norm','Sel. threshold','Supp.'))

  # OMP Main Loop 
  Finished = False
  cdef long count = 0 # Init Counter
  cdef double minepsilon = 0.0
  cdef long selectedCol = 0
  while (not(Finished)):

    # Init counters
    count += 1
    
    # Start with the Norm of b
    minepsilon = np.linalg.norm(b, ord=2)**2

    tmp = np.zeros(A.imShape,dtype=complex)
    tmp[A.samplingSet] = curr_res[:]

    if(ompMethod == 'omp'):

      if(useNorms):
        # Compute vector e
        e = np.absolute((np.linalg.norm(tmp)**2 - (A.adjoint(tmp)**2/Anorm.reshape(A.wavShape))).flatten())
        # Get the minimum of e restricted to the indices not in the support
        e[indexSet] = np.inf
        selectedCol = np.argmin(e)
      else:
        # selectedCol = notIndexSet[np.argmax(np.absolute(A.T * curr_res)[notIndexSet])]
        # Get index with maximum correlation
        # Select a single column       
        selectedCol = notIndexSet[np.argmax(np.absolute(A.adjoint(tmp).flatten())[notIndexSet])]              

      # Add index to index set
      indexSet.append(selectedCol)
      indexSet.sort()
      # Remove from not index set
      notIndexSet.remove(selectedCol)
      notIndexSet.sort()

    elif(ompMethod == 'stomp'):
      
      # Set threshold as per D.L. Donoho et al. 2012 paper 
      thr = np.linalg.norm(curr_res)/np.sqrt(A.samplingSet.sum())*ts_factor
      
      # Compute the Coefficients      
      if(useNorms):
        matchCoeffs = np.absolute((A.adjoint(tmp)/np.sqrt(Anorm.reshape(A.wavShape))).flatten())
      else:
        matchCoeffs = np.absolute(A.adjoint(tmp).flatten())

      for loopA in range(len(matchCoeffs)):
        if(matchCoeffs[loopA] >= thr):
          if not(loopA in indexSet):
            indexSet.append(loopA)
            notIndexSet.remove(loopA)

      # Check if IndexSet is still empty
      # This happens, e.g., with zero velocity components
      if (len(indexSet)==0):
        selectedCol = np.argmax(matchCoeffs)
        indexSet.append(selectedCol)
        notIndexSet.remove(selectedCol)

      # Sort Index Sets
      indexSet.sort()
      notIndexSet.sort()

    else:
      raise ValueError('ERROR: Invalid OMP algorithm.')

    # Initialize lsQR
    B = A.colRestrict(indexSet)
    
    # Solve least Squares    
    lsqr = lsQR(B)    
    lsqr.solve(vec_b)
    ompSol[indexSet] = lsqr.x

    # Update residual
    curr_res = vec_b - A.eval(ompSol.reshape(A.wavShape))[A.samplingSet]

    # Compute relative residual norm
    resNorm = np.linalg.norm(curr_res, ord = 2)/iniResNorm

    # Print Message
    if(showProgress):
      if(count % progressInt == 0)and(count >= progressInt):

        if(ompMethod == 'omp'):
          print('%10d %15e' % (count,resNorm))
        elif(ompMethod == 'stomp'):
          print('%10d %15e %15.1f %8d' % (count,resNorm,thr,len(indexSet)))
    
    # Check finished
    if (resNorm < tol or count > maxItns):
      Finished = True
    
  # Return Result
  return ompSol.reshape(A.wavShape), np.linalg.norm(ompSol.ravel(),1)
