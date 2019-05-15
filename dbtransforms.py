import numpy as np

factor=118
#not sure how to get this values in general, so I specify these for 256x256 image
lenarr = [1,3,3,3,3,3]
sizearr= [14,14,22,38,69,131]

def pywt2array( x, shape ):
    #essentially a breadth-first traversal
    y = np.zeros((0,1))
    for sublist in x:
        for z in sublist:
            y = np.concatenate((y, np.expand_dims(z.ravel(), axis=1)),axis=0)
    y = y.reshape((factor, int(y.shape[0]/factor)))
    return y

def array2pywt( x ):
    shape = x.shape
    #assuming both dims power of 2 or one dim is multiple of other (which is power of 2)
    x = x.ravel()
    coeffs = [None]*len(lenarr)
    i = len(x)
    initdim1=sizearr[0]
    initdim2=initdim1
    index=0
    coeffs[0] = np.reshape(x[index:initdim1*initdim2+index],(initdim1,initdim2))
    index += initdim1*initdim2
    for k in range(1,len(lenarr)):
        coeffs[k] = [None]*lenarr[k]
        for n in range(lenarr[k]):
            coeffs[k][n] = np.reshape(x[index:sizearr[k]*sizearr[k]+index],(sizearr[k],sizearr[k]))
            index += sizearr[k]*sizearr[k]
    return coeffs
