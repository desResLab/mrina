import matplotlib.pyplot as plt
import sys
import os
import cv2
import numpy as np
sys.path.append('../../')
from recover import recover_vel, linear_reconstruction
home = os.getenv("HOME")
    
def save_mask(tosavedir, p, uType, numRealizations, ext='.png'):
    undfile = tosavedir + 'undersamplpattern_p' + str(int(p*100)) + uType + '_n' + str(numRealizations)       
    mask = np.load(undfile + '.npy')
    cv2.imwrite(undfile + ext, np.moveaxis(mask,0,2).astype(int))

def save_cs(noise_percent, p, samptype, recdir, venc, num_samples, ext='.png'):
    recnpy = recdir + 'rec_noise'+str(int(noise_percent*100))+ '_p' + str(int(p*100)) + samptype + '_n' + str(num_samples) + '.npy' 
    recovered = np.load(recnpy)
    imgs = recover_vel(recovered, venc)
    directory = recdir + 'noise' + str(int(noise_percent*100)) 
    if not os.path.exists(directory):
        os.makedirs(directory)
    directory = directory + '/p' + str(int(p*100)) + samptype 
    if not os.path.exists(directory):
        os.makedirs(directory)
    for n in range(imgs.shape[0]):
        for k in range(4):        
           cv2.imwrite(directory + '/rec' + '_n' + str(n) + '_k' + str(k) + ext, imgs[n,k,0])

def save_noisy(noise_percent, recdir, venc, num_samples, ext='.png'):
    noisy = np.load(recdir + 'noisy_noise' + str(int(noise_percent*100)) + '_n' + str(num_samples) + '.npy')
    noisy = linear_reconstruction(noisy)
    imgs = recover_vel(noisy, venc)
    directory = recdir + 'noise' + str(int(noise_percent*100))
    if not os.path.exists(directory):
        os.makedirs(directory)
    for n in range(imgs.shape[0]):
        for k in range(4):        
           cv2.imwrite(directory + '/noisy' + '_n' + str(n) + '_k' + str(k) + ext, imgs[n,k,0])

def save_truth(dir, ext='.png'):
    orig_file = dir+'imgs_n1' +  '.npy'
    true = np.load(orig_file)
    for k in range(4):
        cv2.imwrite(dir + '/true' '_k' + str(k) + ext, true[0,k,0])

def save_all(dir, recdir, patterndir, numRealizations=100):
    save_truth(dir)
    venc = np.load(dir + 'venc_n1' + '.npy')
    for p in [0.25]:#, 0.5, 0.75]:
        for samptype in ['bernoulli']:#, 'vardengauss']:
            save_mask(patterndir, p, samptype, numRealizations) 
            for noise_percent in [0.01]:#, 0.05, 0.1, 0.3]:
                save_cs(noise_percent, p, samptype, recdir, venc, numRealizations)
            
    for noise_percent in [0.01]:#, 0.05, 0.1, 0.3]:
        save_noisy(noise_percent, recdir, venc, numRealizations)  

if __name__ == '__main__':
    if len(sys.argv) == 4:
        dir = sys.argv[1]
        recdir = sys.argv[2]
        patterndir = sys.argv[3]
    elif len(sys.argv) == 2:
        dir = sys.argv[1]
        recdir = dir
        patterndir = dir
    elif len(sys.argv) == 1:
        dir = home + '/apps/undersampled/poiseuille/npy/'#where the kspace data is
        recdir = dir #where to save recovered imgs
        patterndir = home + '/apps/undersampled/poiseuille/npy/' #where the undersampling patterns are located
    save_all(dir, recdir, patterndir)
