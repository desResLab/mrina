import numpy as np
import matplotlib.pyplot as plt

testMat = np.array([[1,2,3],
	                [4,5,6],
	                [7,8,9]])

testMask = np.array([[False,True,False],
	                 [False,False,True],
	                 [True,False,False]])

print(testMat[testMask])



