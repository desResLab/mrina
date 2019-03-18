import os
import random
import numpy as np
import matplotlib.pyplot as plt
import cv2
home = os.getenv('HOME')
num_samples = 100
noise_percent = 0.1
dir = home + "/Documents/undersampled/npy/"
EPSILON = 0.5
plot = False
def circle(shape, center, dist):
    #probably a more efficient way to determine this
    #should cut down x,y range iterates at least
    pts = []
    for x in range(shape[0]):
        for y in range(shape[1]):
            if abs((x-center[0])**2 + (y-center[1])**2- dist**2) < EPSILON**2:
                pts.append([x, y])
    return pts
samples = np.load(dir + 'ndrec_noise'+str(int(noise_percent*100))+'_n'+str(num_samples) + '.npy')
print(len(samples))
samples = np.concatenate(samples, axis=0)
if plot:
    plt.figure()
    plt.imshow(np.abs(samples[0]), cmap='gray')
    plt.title('recovered 1')
    plt.draw()
    plt.figure()
    plt.imshow(np.abs(samples[1]), cmap='gray')
    plt.title('recovered 2')
    plt.draw()
    plt.figure()
    plt.imshow(np.abs(samples[1]-samples[0]), cmap='gray')
    plt.title('diff between recovered')
    plt.draw()
    actual   = cv2.imread('nd_small.jpg', cv2.IMREAD_GRAYSCALE)
    plt.figure()
    plt.imshow(np.abs(actual-samples[0]), cmap='gray')
    plt.title('diff between actual')
    plt.draw()
print(samples.shape)
size = 150
coeff = np.zeros(size)
for k in range(1, size+1):
    pt1 = (random.randint(0, samples.shape[1]-1), random.randint(0, samples.shape[2]-1))
    dir = (random.choice([-1,1]), random.choice([-1,1]))
    #need some check to make sure pt1 isn't situated so that no other pt can be found
    # (for larger distances)
    pt2 = random.choice(circle(samples.shape[1:], pt1, k))
    if k % 50 == 0:
        pts = np.asarray(circle(samples.shape[1:], pt1, k))
        plt.figure()
        plt.scatter(pts[:,0], pts[:,1])
        plt.scatter(pt1[0], pt1[1], c='r')
        plt.title('dist' + str(k))
        plt.draw()
    var1 = np.abs(samples[:,pt1[0], pt1[1]])
    var2 = np.abs(samples[:,pt2[0], pt2[1]])
    rho = np.corrcoef(np.asarray([var1, var2]))
    coeff[k-1] = rho[0,1]
plt.figure()
plt.plot(range(1,size+1), np.abs(coeff))
plt.xlabel('distance')
plt.ylabel('abs of correlation coefficient')
plt.show()
