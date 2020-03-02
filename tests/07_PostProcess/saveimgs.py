import matplotlib.pyplot as plt
import sys
import os
import cv2
import numpy as np
sys.path.append('../../')
#from recuniq import recover_vel, linear_reconstruction, solver_folder
from recover import recover_vel, linear_reconstruction, solver_folder
home = os.getenv("HOME")

def rescale(img, newmax=255, truncate=True, truncateMax=1):
    if truncate:
      img[img<0] = 0
      img[img>truncateMax] = truncateMax
    newimg = np.zeros(img.shape)
    for n in range(img.shape[0]):
        for k in range(img.shape[1]):
            minimum = np.amin(img[n,k])
            maximum = np.amax(img[n,k])
            if (maximum-minimum > 1E-3):
                newimg[n,k] = (img[n,k]-minimum)*(newmax/(maximum - minimum))
    print('new', np.amin(newimg), np.amax(newimg))
    return newimg

def pltimg(img, title):
  if len(img.shape)>2:
    img = img[0]
  plt.figure()
  plt.imshow(np.real(img))
  plt.colorbar()
  plt.title(title)
  plt.draw()

def save_mask(tosavedir, p, uType, numRealizations, relative, ext='.png'):
    undfile = tosavedir + 'undersamplpattern_p' + str(int(p*100)) + uType + '_n' + str(numRealizations)       
    mask = np.load(undfile + '.npy')
    mask = mask.astype(int)
    if relative:
        mask = rescale(mask)
    if len(mask.shape) == 2:
        mask = np.expand_dims(mask,0)
    cv2.imwrite(undfile + ext, np.moveaxis(mask,0,2))

def save_cs(noise_percent, p, samptype, recdir, venc, num_samples, relative, ext='.png'):
    recnpy = recdir + 'rec_noise'+str(int(noise_percent*100))+ '_p' + str(int(p*100)) + samptype + '_n' + str(num_samples) + '.npy' 
    recovered = np.load(recnpy)
    imgs = recover_vel(recovered, venc)
    #pltimg(imgs[0,1], 'recovered before rescale')
    if relative:
        imgs = rescale(imgs)
    pltimg(imgs[0,1], 'recovered after rescale')
    directory = recdir + 'noise' + str(int(noise_percent*100)) + '/p' + str(int(p*100)) + samptype 
    if not os.path.exists(directory):
        os.makedirs(directory)
    for n in range(1):#range(imgs.shape[0]):
        for k in range(imgs.shape[1]):
           cv2.imwrite(directory + '/rec_p' + str(int(p*100)) + '_n' + str(n) + '_k' + str(k) + ext, imgs[n,k,0])
           #cv2.imwrite(directory + '/rec' + '_n' + str(n) + '_k' + str(k) + ext, imgs[n,k,0])

def save_rec_noise(noise_percent, p, samptype, dir, recdir, venc, num_samples, relative, use_truth=False, ext='.png'):
    recnpy = recdir + 'rec_noise'+str(int(noise_percent*100))+ '_p' + str(int(p*100)) + samptype + '_n' + str(num_samples) + '.npy' 
    recovered = np.load(recnpy)
    orig_file = dir+'imgs_n1' +  '.npy'
    true = np.load(orig_file)
    imgs = recover_vel(recovered, venc)
    avg = imgs.mean(axis=0)
    for i in range(imgs.shape[0]):
        for j in range(imgs.shape[1]):
            for k in range(imgs.shape[2]):
                if use_truth:
                    imgs[i,j,k] = imgs[i,j,k] - true[0,j,k]    
                else:
                    imgs[i,j,k] = imgs[i,j,k] - avg[j,k] 
    if use_truth:
        desc = 'true'
    else:
        desc = 'avg'
    print(desc + " MSE ", ((imgs)**2).mean(axis=None))
    if relative:
        imgs = rescale(imgs, truncate=False)
    directory = recdir + 'noise' + str(int(noise_percent*100)) + '/p' + str(int(p*100)) + samptype 
    if not os.path.exists(directory):
        os.makedirs(directory)
    for n in range(1): #range(imgs.shape[0]):
        for k in range(imgs.shape[1]):        
            cv2.imwrite(directory + '/' + desc + 'recnoise' + '_n' + str(n) + '_k' + str(k) + ext, imgs[n,k,0])

def save_noisy(noise_percent, dir, recdir, venc, num_samples, relative, ext='.png'):
    noisy = np.load(dir + 'noisy_noise' + str(int(noise_percent*100)) + '_n' + str(num_samples) + '.npy')
    noisy = linear_reconstruction(noisy)
    avg = noisy.mean(axis=0)
    orig_file = dir+'imgs_n1' +  '.npy'
    imgs = recover_vel(noisy, venc)
    if relative:
        imgs = rescale(imgs)
    directory = recdir + 'noise' + str(int(noise_percent*100))
    if not os.path.exists(directory):
        os.makedirs(directory)
    for n in range(5):#range(imgs.shape[0]):
        for k in range(imgs.shape[1]):    
            cv2.imwrite(directory + '/noisy' + '_n' + str(n) + '_k' + str(k) + ext, imgs[n,k,0])

def save_truth(dir, relative, ext='.png'):
    orig_file = dir+'imgs_n1' +  '.npy'
    true = np.load(orig_file)
    if relative:
        true = rescale(true, truncate=False)
    print('save truth')
    for k in range(true.shape[1]):
        cv2.imwrite(dir + '/true' '_k' + str(k) + ext, true[0,k,0])

def save_all(dir, recdir, patterndir, numRealizations=100, relative=True):
    save_truth(dir, relative)
    vencfile = dir + 'venc_n1.npy'
    if os.path.exists(vencfile):
        venc = np.load(dir + 'venc_n1' + '.npy')
    else:
        venc = None
    for p in [0.25, 0.5, 0.75]:
        for samptype in ['bernoulli', 'vardengauss']:
            save_mask(patterndir, p, samptype, numRealizations, relative) 
            for noise_percent in [0.01, 0.05, 0.1, 0.3]:
                save_cs(noise_percent, p, samptype, recdir, venc, numRealizations, relative)
            
    for noise_percent in [0.01, 0.05, 0.1, 0.3]:
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
    #save_all(dir, recdir, patterndir, numRealizations)
    #save_mask(patterndir, p, samptype, numRealizations, relative) 
    noise_percent=0.1
    p=0.75
    venc = np.load(dir+ "venc_n1.npy")#None
    relative=False
    #relative=True
    #save_truth(dir, relative)
    nngpfile = np.loadtxt(home + '/apps/undersampled/nngp/repo/nnGP/data/10_aorta/y.mod')
    nngpfile = np.reshape(nngpfile, (256,256))
    for samptype in ['vardengauss']:#['bernoulli', 'vardengauss']:
        save_cs(noise_percent, p, samptype, recdir, venc, numRealizations, relative)
        save_rec_noise(noise_percent, p, samptype, dir, recdir, venc, numRealizations, relative, use_truth=False)
        save_rec_noise(noise_percent, p, samptype, dir, recdir, venc, numRealizations, relative, use_truth=True)
        save_noisy(noise_percent, dir, recdir, venc, numRealizations, relative)
