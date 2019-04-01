import os
import random
import numpy as np
import matplotlib.pyplot as plt
import cv2
home = os.getenv('HOME')
num_samples = 100 #number of samples in file
noise_percent = 0.1
dir = home + "/Documents/undersampled/npy/"
EPSILON = 0.5
n=100 #number of samples to include
plot = False

fs=8
plt.rc('font',  family='serif')
plt.rc('xtick', labelsize='x-small')
plt.rc('ytick', labelsize='x-small')
plt.rc('text',  usetex=True)

def circle(shape, center, dist):
    #probably a more efficient way to determine this
    pts = []
    for x in range(max(0,center[0]-dist), min(center[0]+dist, shape[0])):#shape[0]):
        for y in range(max(0,center[1]-dist), min(center[1]+dist, shape[1])):
            if abs((x-center[0])**2 + (y-center[1])**2- dist**2) < EPSILON**2:
                pts.append([x, y])
    return pts
samples = np.load(dir + 'ndrec_noise'+str(int(noise_percent*100))+'_n'+str(num_samples) + '.npy')
samples = np.concatenate(samples, axis=0)
samples = samples[0:n]
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

def select_points(dist):
    #todo; make pt1 selection avoid the while loop case
    #i.e. only select from applicable points for pt1
    pt1 = (random.randint(0, samples.shape[1]-1), random.randint(0, samples.shape[2]-1))
    choices = circle(samples.shape[1:], pt1, k)
    while not choices: #make sure there were points within range for distance
        pt1 = (random.randint(0, samples.shape[1]-1), random.randint(0, samples.shape[2]-1))
        choices = circle(samples.shape[1:], pt1, dist)
    pt2 = random.choice(choices)
    return pt1,pt2
print(samples.shape)
size = 200
num_pts = 50
coeff = np.zeros((size, num_pts))
corravg = np.zeros(size)
for k in range(1, size+1):
    points = [None]*num_pts
    for j in range(num_pts):
        pt1, pt2 = select_points(k)
        while (pt1,pt2) in points: #select new points if we've already looked at these
            pt1, pt2 = select_points(k) #todo; need to recognize order of points is irrelevant
        points[j] = (pt1,pt2)
        if k % 50 == 0 and j == 1:
            pts = np.asarray(circle(samples.shape[1:], pt1, k))
            plt.figure()
            plt.scatter(pts[:,0], pts[:,1])
            plt.scatter(pt1[0], pt1[1], c='r')
            plt.xlim(0,256)
            plt.ylim(0,256)
            plt.title('dist' + str(k))
            plt.draw()
        var1 = np.abs(samples[:,pt1[0], pt1[1]])
        var2 = np.abs(samples[:,pt2[0], pt2[1]])
        coeff[k-1, j] = np.corrcoef(np.asarray([var1, var2]))[0,1]**2#R^2 correlation
    corravg[k-1] = np.mean(coeff[k-1])
plt.figure(figsize=(4,3))
plt.plot(range(1,size+1), np.abs(corravg))
plt.xlabel('Distance',fontsize=fs)
plt.ylabel('Correlation Coefficient ($R^2$)',fontsize=fs)
plt.tick_params(labelsize=fs)
plt.savefig('results/corrsqravg' + str(num_pts) + '_noise' + str(noise_percent*100) + '_n'+str(n) + '.png')
#plt.draw()
np.save('results/corrsqravg' + str(num_pts) + '_noise' + str(noise_percent*100) + '_n'+str(n), coeff)
spc = 5
plt.figure(figsize=(4,3))
plt.boxplot(np.abs(coeff[::spc]).T, sym='.')
sz = len(coeff[::spc])
plt.xticks(range(sz+1),range(0, size+1, spc))
ax = plt.axes()
for index, label in enumerate(ax.xaxis.get_ticklabels()):
    if index % 4 != 0 and index!=sz:
        label.set_visible(False)
plt.xlabel('Distance',fontsize=fs)
plt.ylabel('Correlation Coefficient ($R^2$)',fontsize=fs)
plt.tick_params(labelsize=fs)
plt.savefig('results/corrsqrboxplt' + str(num_pts) + '_noise' + str(noise_percent*100) + '_n'+str(n) + '.png')
#plt.show()
