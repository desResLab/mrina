import os
import random
import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys
sys.path.append('../../')
from recover import recover_vel
home = os.getenv('HOME')
num_samples = 100 #number of samples in file
noise_percent = 0.01
p= 0.5
type='bernoulli'
#dir = home + "/Documents/undersampled/npy/"
dir = home + "/apps/undersampled/modelflow/aorta_orig/npy/"
EPSILON = 0.5
n=100 #number of samples to include
plotExamples = False 

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

def select_points(dist, imsz):
    #todo; make pt1 selection avoid the while loop case
    #i.e. only select from applicable points for pt1
    pt1 = (random.randint(0, imsz[0]-1), random.randint(0, imsz[1]-1))
    choices = circle(imsz, pt1, dist)
    while not choices: #make sure there were points within range for distance
        pt1 = (random.randint(0, imsz[0]-1), random.randint(0, imsz[1]-1))
        choices = circle(imsz, pt1, dist)
    pt2 = tuple(random.choice(choices))
    if pt1 <= pt2:#pt1[0] < pt2[0] or (pt1[0] == pt2[0] and pt1[1] <= pt2[1]):
        return pt1,pt2
    return pt2, pt1

def get_points(size, num_pts,imsz):
    print('getting points...')
    points = np.zeros((size, num_pts, 2, 2), dtype=int)
    for k in range(1,size+1):
        print(k)
        for j in range(num_pts):
            pt1, pt2 = select_points(k,imsz)
            while pt1[0] in points[k-1,:,0,0] and pt1[1] in points[k-1,:,0,1] and pt2[0] in points[k-1,:,1,0] and pt2[0] in points[k-1,:,1,1]:
                #select new points if we've already selected these
                pt1, pt2 = select_points(k, imsz) 
            points[k-1, j, 0, 0] = pt1[0]
            points[k-1, j, 0, 1] = pt1[1]
            points[k-1, j, 1, 0] = pt2[0]
            points[k-1, j, 1, 1] = pt2[1]
    np.save(dir+'points_s' + str(size) + '_n' + str(num_pts),points)

def get_samples(noise_percent, p, num_samples):
    samples = np.load(dir + 'rec_noise'+str(int(noise_percent*100))+'_p' + str(int(p*100)) + type +'_n'+str(num_samples) + '.npy')
    venc = np.load(dir + 'venc_n1' + '.npy')
    samples = recover_vel(samples, venc)
    print(samples.shape)
    samples = np.squeeze(samples)
    print(samples.shape)
    samples = samples[0:n]
    return samples

def plot(samples):
    plt.figure()
    plt.imshow(np.abs(samples[0,3]), cmap='gray')
    plt.title('recovered 1')
    plt.colorbar()
    plt.draw()
    plt.figure()
    plt.imshow(np.abs(samples[1,3]), cmap='gray')
    plt.title('recovered 2')
    plt.colorbar()
    plt.draw()
    plt.figure()
    plt.imshow(np.abs(samples[1,3]-samples[0,3,0]), cmap='gray')
    plt.title('diff between recovered')
    plt.colorbar()
    plt.draw()
    #actual   = cv2.imread('nd_small.jpg', cv2.IMREAD_GRAYSCALE)
    actual   = np.load(dir + 'imgs_n1.npy')
    print(actual.shape)
    plt.figure()
    plt.imshow(np.abs(actual[0,3,0]), cmap='gray')
    plt.title('actual')
    plt.colorbar()
    plt.draw()
    plt.figure()
    plt.imshow(np.abs(actual[0,3,0]-samples[0,3]), cmap='gray')
    plt.title('diff between actual')
    plt.colorbar()
    plt.show()

def get_saved_points(samples, size, num_pts):
    imsz = samples.shape[2:]
    if not os.path.isfile(dir+'points_s' + str(size) + '_n' + str(num_pts) + '.npy'):
        get_points(size,num_pts,imsz)
    points = np.load(dir+'points_s' + str(size) + '_n' + str(num_pts) + '.npy')
    return points

def get_coeff(size, num_pts, samples, points):
    coeff = np.zeros((4, size, num_pts))
    corravg = np.zeros((4,size))

    for k in range(1, size+1):
        for j in range(num_pts):
            pt1 = (points[k-1, j, 0,0], points[k-1,j, 0,1])
            pt2 = (points[k-1, j, 1,0], points[k-1,j, 1,1])
            #print(pt1,pt2)
            for v in range(0,4):
                var1 = np.abs(samples[:, v,pt1[0], pt1[1]])
                var2 = np.abs(samples[:, v,pt2[0], pt2[1]])
                coeff[v, k-1, j] = np.corrcoef(np.asarray([var1, var2]))[0,1]**2#R^2 correlation
        for v in range(0,4):
            corravg[v, k-1] = np.mean(coeff[v, k-1])
    return coeff

def get_vals(noise_percent, p, num_samples, size, num_pts):
    samples = get_samples(noise_percent, p, num_samples)
    points = get_saved_points(samples, size, num_pts)
    coeff = get_coeff(size, num_pts, samples, points)
    np.save(dir + 'results/corrsqravg' + str(num_pts) + '_noise' + str(int(noise_percent*100)) + '_p' + str(int(p*100)) + type +'_n'+str(n), coeff)
    return coeff

size = 100
num_pts = 50

def get_all():
    for noise_percent in [0.01, 0.05, 0.1, 0.3]:
        for p in [0.25, 0.5, 0.75]:
            print('noise %', noise_percent, 'undersampling prob', p)
            get_vals(noise_percent, p, 100, size, num_pts)

if plotExamples:
    plot(get_samples(noise_percent, p, num_samples))

get_all()
coeff = get_vals(noise_percent, p, num_samples, size, num_pts)
corravg = np.mean(coeff[0], axis=1)

plt.figure(figsize=(4,3))
plt.plot(range(1,size+1), np.abs(corravg[0]))
plt.xlabel('Distance',fontsize=fs)
plt.ylabel('Correlation Coefficient ($R^2$)',fontsize=fs)
plt.tick_params(labelsize=fs)
#plt.savefig(dir + 'results/corrsqravg' + str(num_pts) + '_noise' + str(int(noise_percent*100)) + '_n'+str(n) + '.png')
plt.draw()

spc = 5
plt.figure(figsize=(4,3))
plt.boxplot(np.abs(coeff[0][::spc]).T, sym='.')
sz = len(coeff[0][::spc])
plt.xticks(range(sz+1),range(0, size+1, spc))
ax = plt.axes()
for index, label in enumerate(ax.xaxis.get_ticklabels()):
    if index % 4 != 0 and index!=sz:
        label.set_visible(False)
plt.xlabel('Distance',fontsize=fs)
plt.ylabel('Correlation Coefficient ($R^2$)',fontsize=fs)
plt.tick_params(labelsize=fs)
#plt.savefig(dir + 'results/corrsqrboxplt' + str(num_pts) + '_noise' + str(int(noise_percent*100)) + '_n'+str(n) + '.png')
plt.show()
