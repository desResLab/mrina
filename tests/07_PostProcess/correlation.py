import os
import random
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../../')
from genSamples import recover_vel
home = os.getenv('HOME')
num_samples = 100 #number of samples in file
noise_percent = 0.01
p= 0.5
type='bernoulli'
recdir = home + "/apps/undersampled/modelflow/aorta_orig/npy/" #where recovered images are
ptsdir = home + "/apps/undersampled/poiseuille/npy/" #where location of points chosen are stored
kspacedir = ptsdir #where noisy kspace 
EPSILON = 0.5
n=100 #number of samples to include

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
    print('getting points of size ' + str(size) + ' and ' + str(num_pts) + ' points...')
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
    np.save(ptsdir+'points_s' + str(size) + '_n' + str(num_pts),points)

def get_samples(noise_percent, p, samptype, num_samples, recdir, kspacedir):
    samples = np.load(recdir + 'rec_noise'+str(int(noise_percent*100))+'_p' + str(int(p*100)) + samptype +'_n'+str(num_samples) + '.npy')
    venc = np.load(kspacedir + 'venc_n1' + '.npy')
    samples = recover_vel(samples, venc)
    print(samples.shape)
    samples = np.squeeze(samples)
    print(samples.shape)
    samples = samples[0:n]
    return samples

def get_saved_points(samples, size, num_pts, ptsdir):
    imsz = samples.shape[2:]
    if not os.path.isfile(ptsdir+'points_s' + str(size) + '_n' + str(num_pts) + '.npy'):
        get_points(size,num_pts,imsz)
    points = np.load(ptsdir+'points_s' + str(size) + '_n' + str(num_pts) + '.npy')
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

def get_vals(noise_percent, p, samptype, num_samples, size, num_pts, save_numpy=True, recdir=recdir, kspacedir=kspacedir, ptsdir=ptsdir):
    samples = get_samples(noise_percent, p, samptype, num_samples, recdir, kspacedir)
    points = get_saved_points(samples, size, num_pts, ptsdir)
    coeff = get_coeff(size, num_pts, samples, points)
    if save_numpy:
        np.save(recdir + 'results/corrsqravg' + str(num_pts) + '_noise' + str(int(noise_percent*100)) + '_p' + str(int(p*100)) + type +'_n'+str(n), coeff)
    return coeff

def get_all(size, num_pts):
    #size is max. distance to retrieve correlations for
    #num_pts is the number of points to average correlation across
    for noise_percent in [0.01, 0.05, 0.1, 0.3]:
        for p in [0.25, 0.5, 0.75]:
            for samptype in ['bernoulli', 'vardengauss']:
                try:
                    get_vals(noise_percent, p, samptype, 100, size, num_pts)
                    print('saved noise %', noise_percent, 'undersampling prob', p, 'sampling type', samptype)
                except Exception as e:
                    print(e)
                    print('may be missing noise %', noise_percent, 'undersampling prob', p, 'sampling type', samptype)
                    continue

if __name__ == '__main__':
    if len(sys.argv) > 1:
        size = int(sys.argv[1])
        num_pts=int(sys.argv[2])
    else:
        size = 100 #max dist from point
        num_pts = 50 # number of points at each distance
    #to save only one correlation average:
    #get_vals(noise_percent, p, samptype, num_samples, size, num_pts)
    #calculate correlations for all noise levels, undersampling levels, and sampling types bernoulli and vardengauss
    # and save to numpy files
    get_all(size, num_pts)
