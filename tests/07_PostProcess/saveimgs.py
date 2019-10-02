import matplotlib.pyplot as plt
import sys
import os
import cv2
import numpy as np
sys.path.append('../../')
from recover import recover_vel, linear_reconstruction, solver_folder
home = os.getenv("HOME")

def rescale(img):
    #img[img<0] = 0
    minimum = np.amin(img)
    return 255*(img-minimum)/(np.amax(img) - minimum)

def save_mask(tosavedir, p, uType, numRealizations, relative, ext='.png'):
    undfile = tosavedir + 'undersamplpattern_p' + str(int(p*100)) + uType + '_n' + str(numRealizations)       
    mask = np.load(undfile + '.npy')
    mask = mask.astype(int)
    if relative:
        mask = rescale(mask)
    cv2.imwrite(undfile + ext, np.moveaxis(mask,0,2).astype(int))

def save_cs(noise_percent, p, samptype, recdir, venc, num_samples, relative, ext='.png'):
    recnpy = recdir + 'rec_noise'+str(int(noise_percent*100))+ '_p' + str(int(p*100)) + samptype + '_n' + str(num_samples) + '.npy' 
    recovered = np.load(recnpy)
    imgs = recover_vel(recovered, venc)
    if relative:
        imgs = rescale(imgs)
    directory = recdir + 'noise' + str(int(noise_percent*100)) 
    if not os.path.exists(directory):
        os.makedirs(directory)
    directory = directory + '/p' + str(int(p*100)) + samptype 
    if not os.path.exists(directory):
        os.makedirs(directory)
    for n in range(imgs.shape[0]):
        for k in range(imgs.shape[1]):        
           cv2.imwrite(directory + '/rec' + '_n' + str(n) + '_k' + str(k) + ext, imgs[n,k,0])

def save_noisy(noise_percent, dir, recdir, venc, num_samples, relative, ext='.png'):
    noisy = np.load(dir + 'noisy_noise' + str(int(noise_percent*100)) + '_n' + str(num_samples) + '.npy')
    noisy = linear_reconstruction(noisy)
    imgs = recover_vel(noisy, venc)
    if relative:
        imgs = rescale(imgs)
    directory = recdir + 'noise' + str(int(noise_percent*100))
    if not os.path.exists(directory):
        os.makedirs(directory)
    for n in range(imgs.shape[0]):
        for k in range(imgs.shape[1]):        
           cv2.imwrite(directory + '/noisy' + '_n' + str(n) + '_k' + str(k) + ext, imgs[n,k,0])

def save_truth(dir, relative, ext='.png'):
    orig_file = dir+'imgs_n1' +  '.npy'
    true = np.load(orig_file)
    if relative:
        true = rescale(true)
    for k in range(true.shape[1]):
        cv2.imwrite(dir + '/true' '_k' + str(k) + ext, true[0,k,0])

def save_all(dir, recdir, patterndir, numRealizations=100, relative=True):
    save_truth(dir, relative)
    venc = np.load(dir + 'venc_n1' + '.npy')
    for p in [0.25, 0.5, 0.75]:
        for samptype in ['bernoulli', 'vardengauss']:
            save_mask(patterndir, p, samptype, numRealizations, relative) 
            for noise_percent in [0.01, 0.05, 0.1, 0.3]:
                save_cs(noise_percent, p, samptype, recdir, venc, numRealizations, relative)
            
    for noise_percent in [0.01]:#, 0.05, 0.1, 0.3]:
        save_noisy(noise_percent, dir, recdir, venc, numRealizations, relative)  

if __name__ == '__main__':
    if len(sys.argv) == 6:
        numRealizations = int(sys.argv[1])
        dir = sys.argv[2]
        recdir = sys.argv[3]
        patterndir = sys.argv[4]
        solver_mode = int(sys.argv[5])
    elif len(sys.argv) == 3:
        numRealizations = int(sys.argv[1])
        dir = sys.argv[2]
        recdir = dir
        patterndir = dir
        solver_mode = 0
    elif len(sys.argv) == 1:
        numRealizations = 100
        dir = home + '/apps/undersampled/poiseuille/npy/'#where the kspace data is
        recdir = dir #where to save recovered imgs
        patterndir = home + '/apps/undersampled/poiseuille/npy/' #where the undersampling patterns are located
        solver_mode = 0
    recdir = recdir + "/" + solver_folder(solver_mode) + "/"
    save_all(dir, recdir, patterndir, numRealizations)
