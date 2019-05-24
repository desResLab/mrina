import sys
import numpy as np
from CSRecoverySuite import OMPRecovery
from CSRecoverySuite import OperatorLinear
from lsqr            import lsQR

def linearTest():
  # Solve the SQD system
  #  [ 2  1 ] [x] = [2]
  #  [ 1 -3 ] [y]   [0]

  m = 100 # Number of Rows
  n = 10  # Number of Columns
  aMat = np.random.randn(m,n)
  sol  = np.random.randn(n)
  rhs  = np.dot(aMat,sol)
  
  # Initialize the operator
  A = OperatorLinear(aMat)
  
  # Init lsQR
  lsqr = lsQR(A)
  
  # Solve the least-square problem
  lsqr.solve(rhs, show=True)
  for loopA in range(n):
    print('%f %f' % (sol[loopA],lsqr.x[loopA]))

def ompTest():

  m = 50
  n = 200
  p = 10

  np.random.seed(1244)
  aMat = np.random.randn(m,n)
  x = np.zeros(n)
  index_set = np.random.randint(0, 10, size=p)
  x[index_set] = np.random.normal(loc = 6, scale = 1, size = p)
  b = aMat.dot(x)

  # Initialize the operator
  A = OperatorLinear(aMat)

  # Solve with OMP
  ompSol = OMPRecovery(A, b)

  # Print the original and reconstructed solution
  for loopA in range(n):
    print('%f %f' % (x[loopA],ompSol[loopA]))

# MAIN
if __name__ == '__main__':

  # Perform Simple Linear Test
  # linearTest()
  # Perform Test for OMP with linear operators
  ompTest()


