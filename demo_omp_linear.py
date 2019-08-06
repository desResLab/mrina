import sys
import numpy as np
from CSRecoverySuite import OMPRecovery
from CSRecoverySuite import OperatorLinear
from lsqr            import lsQR

def linearTest():
  print('--- TEST FOR LINEAR COMPLEX SYSTEM')
  # Solve the SQD system
  #  [ 2  1 ] [x] = [2]
  #  [ 1 -3 ] [y]   [0]

  m = 100 # Number of Rows
  n = 10  # Number of Columns
  aMat = np.random.randn(m,n) + 1j * np.random.randn(m,n)
  sol  = np.random.randn(n) + 1j * np.random.randn(n)
  rhs  = np.dot(aMat,sol)
  
  # Initialize the operator
  A = OperatorLinear(aMat)
  
  # Init lsQR
  lsqr = lsQR(A)
  
  # Solve the least-square problem with LSQR
  lsqr.solve(rhs, itnlim=100, show=True)

  # Try with the numpy least squares solver
  lsSol = np.linalg.lstsq(aMat, rhs, rcond=None)[0]

  # Plot the Residual
  print('Residual Norm with True Solution: ', np.linalg.norm(np.dot(aMat,sol)-rhs))
  print('Residual Norm LSQR: ', np.linalg.norm(np.dot(aMat,lsqr.x)-rhs))
  print('Residual Norm Numpy LS: ', np.linalg.norm(np.dot(aMat,lsSol)-rhs))
  #for loopA in range(n):
  #  print(sol[loopA], lsqr.x[loopA], lsSol[loopA])

def ompTest():
  print('--- TEST FOR OMP')

  m = 50
  n = 200
  p = 10

  np.random.seed(1344)
  aMat = np.random.randn(m,n) + 1j * np.random.randn(m,n)
  x = np.zeros(n,dtype=np.complex)
  index_set = np.random.randint(0, n, size=p)
  x[index_set] = np.random.normal(loc = 6, scale = 1, size = p) + 1j * np.random.normal(loc = 6, scale = 1, size = p)
  b = np.dot(aMat,x)

  print(x)

  # Initialize the operator
  A = OperatorLinear(aMat)

  # Solve with OMP
  ompSol = OMPRecovery(A, b)[0]

  # Print the original and reconstructed solution
  for loopA in range(n):
    print(x[loopA],ompSol[loopA])

# MAIN
if __name__ == '__main__':

  # Perform Simple Linear Test
  linearTest()
  # Perform Test for OMP with linear operators
  ompTest()


