import sys,os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import cv2
from mrina.recover import recover_vel, linear_reconstruction
import argparse

def pltimg(img, title):
  if len(img.shape)>2:
    img = img[0]
  plt.figure()
  plt.imshow(np.real(img))
  plt.colorbar()
  plt.title(title)
  plt.draw()

def save_mask(maskFile, outputFile):
  '''
  Save undersampling mask to file
  '''
  mask = np.load(maskFile).astype(bool)
  if(len(mask.shape) == 3):
    plt.imshow(np.absolute(np.fft.fftshift(mask[0])), cmap='gray', vmin=0, vmax=1)
  else:
    plt.imshow(np.absolute(np.fft.fftshift(mask)), cmap='gray', vmin=0, vmax=1)
  plt.axis('off')
  plt.savefig(outputFile, bbox_inches='tight', pad_inches=0)
  plt.close()

def save_rec(infilename, venc, p, samptype, noise_percent, wavetype, algtype, prefix, maindir, outputdir, limits, fluidmaskfile, singleChannel=False, relative=False):
  
  recovered = np.load(infilename)

  if(singleChannel):
    imgs = np.absolute(recovered)
  else:
    imgs = recover_vel(recovered, venc)

  # Set to zero the pixels outside the fluid mask
  if(fluidmaskfile != ''):
    print('Using fluid mask: ',maindir + fluidmaskfile)
    fluidmask = np.load(maindir + fluidmaskfile)
    imgs[0,:,0,np.logical_not(fluidmask)] = 0.0
    
  # Only the first reconstructed Sample
  for n in range(1): #range(imgs.shape[0]):
    for k in range(imgs.shape[1]):
      # Custom image Scaling for Comparison
      myvmin=limits[2*k]
      myvmax=limits[2*k+1]
      if(myvmin is not None):
        print('-- Using min value: ',myvmin)
      if(myvmax is not None):
        print('-- Using max value: ',myvmax)

      # Show pictures
      plt.imshow(imgs[n,k,0], cmap='gray',vmin=myvmin,vmax=myvmax)
      plt.axis('off')
      plt.savefig(outputdir + prefix + 'rec_p' + str(int(p*100)) + samptype + \
                                      '_noise' + str(int(noise_percent*100)) + \
                                          '_n' + str(n) + \
                                          '_w' + str(wavetype) + \
                                          '_a' + str(algtype) + \
                                          '_k' + str(k) + '.png', bbox_inches='tight', pad_inches=0)
      plt.close()

def save_lin(infilename,
             venc, p, samptype, noise_percent, wavetype, num_samples,
             prefix, maindir, outputdir, limits, fluidmaskfile, usemultipatterns,
             singleChannel=False, relative=False):

  if(usemultipatterns):
    mask_file = maindir + 'undersamplpattern_p' + str(int(p*100)) + samptype + '_n' + str(num_samples) + '.npy'
  else:
    mask_file = maindir + 'undersamplpattern_p' + str(int(p*100)) + samptype + '.npy'

  mask = np.load(maindir + mask_file)
  noisy = np.load(maindir + 'noisy_noise' + str(int(noise_percent*100)) + '_n' + str(num_samples) + '.npy')
  linrec = linear_reconstruction(noisy,mask)

  if(singleChannel):
    imgs = np.absolute(linrec)
  else:
    imgs = recover_vel(linrec, venc)

  # Set to zero the pixels outside the fluid mask
  if(fluidmaskfile != ''):
    print('Using fluid mask: ',maindir + fluidmaskfile)
    fluidmask = np.load(maindir + fluidmaskfile)
    imgs[0,:,0,np.logical_not(fluidmask)] = 0.0
    
  # Only the first reconstructed Sample
  for n in range(1): #range(imgs.shape[0]):
    for k in range(imgs.shape[1]):
      # Custom image Scaling for Comparison
      myvmin=limits[2*k]
      myvmax=limits[2*k+1]
      if(myvmin is not None):
        print('-- Using min value: ',myvmin)
      if(myvmax is not None):
        print('-- Using max value: ',myvmax)

      # Show pictures
      plt.imshow(imgs[n,k,0], cmap='gray',vmin=myvmin,vmax=myvmax)
      plt.axis('off')
      plt.savefig(outputdir + prefix + 'linrec_p' + str(int(p*100)) + samptype + \
                                      '_noise' + str(int(noise_percent*100)) + \
                                          '_n' + str(n) + \
                                          '_w' + str(wavetype) + \
                                          '_k' + str(k) + '.png', bbox_inches='tight', pad_inches=0)
      plt.close()


def save_rec_noise(recnpyFile, orig_file, venc, p, samptype, noise_percent, wavetype, algtype, prefix, outputdir, singleChannel=False, use_truth=False, ext='.png'):
  
  # Read File with CS Reconstructions 
  recovered = np.load(recnpyFile)
  
  # Read True underlying Image
  true = np.load(orig_file)

  # Check consistency 
  if(recovered.shape[-2:] != true.shape[-2:]):
    print('ERROR: reconstructed images and true images are not compatible')
    sys.exit(-1)

  # Reconstruct velocities from complex recovered images
  if(singleChannel):
    imgs = np.absolute(recovered)
  else:
    imgs = recover_vel(recovered, venc)

  # Compute Average Image
  if(not(use_truth)):
    avg = imgs.mean(axis=0)

  # # DEBUG - CHECK THE DIFFERENCE IN RECONSTRUCTED IMAGE
  # plt.figure()
  # plt.subplot(1,2,1)
  # plt.title('TRUE')
  # plt.imshow(np.absolute(true[0,3,0]))
  # plt.colorbar()
  # plt.subplot(1,2,2)
  # plt.title('REC')
  # plt.imshow(np.absolute(imgs[0,3,0]))
  # plt.colorbar()
  # plt.show()
  # exit(-1)

  # Substract images either from the truth or the average reconstruction
  imgRes = np.zeros_like(imgs)
  for i in range(imgs.shape[0]):
    if use_truth:
      boolMask = np.logical_or((np.absolute(imgs[i,:,:,:,:]) < 1.0e-12),(np.absolute(true[0,:,:,:,:]) < 1.0e-12))
      with np.errstate(divide='ignore', invalid='ignore'):
        imgRes[i,:,:,:,:] = 2*(imgs[i,:,:,:,:] - true[0,:,:,:,:])/(np.absolute(imgs[i,:,:,:,:]) + np.absolute(true[0,:,:,:,:]))      
      imgRes[i,boolMask] = 0.0
    else:
      boolMask = np.logical_or((np.absolute(imgs[i,:,:,:,:]) < 1.0e-12),(np.absolute(true[0,:,:,:,:]) < 1.0e-12))
      with np.errstate(divide='ignore', invalid='ignore'):
        imgRes[i,:,:,:,:] = 2*(imgs[i,:,:,:,:] - avg[:,:,:,:])/(np.absolute(imgs[i,:,:,:,:]) + np.absolute(avg[:,:,:,:]))
      imgRes[i,boolMask] = 0.0
      
  if(use_truth):
    desc = 'true'
  else:
    desc = 'avg'

  for n in range(1): # range(imgs.shape[0]): # Loop on the number of samples
    for k in range(imgRes.shape[1]):
      plt.imshow(imgRes[n,k,0], cmap='seismic', vmin=-2, vmax=2)
      plt.axis('off')
      # plt.show()
      plt.savefig(outputdir + prefix + desc + 'recerror' + '_p' + str(int(p*100)) + samptype + \
                                                       '_noise' + str(int(noise_percent*100)) + \
                                                           '_n' + str(n) + \
                                                           '_w' + str(wavetype) + \
                                                           '_a' + str(algtype) + \
                                                           '_k' + str(k) + ext, bbox_inches='tight', pad_inches=0)
      plt.close()
      
def save_noisy(noise_percent, dir, recdir, venc, num_samples, relative=False, ext='.png'):
    noisy = np.load(dir + 'noisy_noise' + str(int(noise_percent*100)) + '_n' + str(num_samples) + '.npy')
    noisy = linear_reconstruction(noisy)
    avg = noisy.mean(axis=0)
    orig_file = dir+'imgs_n1' +  '.npy'
    imgs = recover_vel(noisy, venc)
    # if relative:
    #     imgs = rescale(imgs)
    directory = recdir + 'noise' + str(int(noise_percent*100))
    if not os.path.exists(directory):
        os.makedirs(directory)
    for n in range(5):#range(imgs.shape[0]):
        for k in range(imgs.shape[1]):    
            cv2.imwrite(directory + '/noisy' + '_n' + str(n) + '_k' + str(k) + ext, imgs[n,k,0])

def save_true(trueFileName, outputDir, relative=False):
  true = np.load(trueFileName)
  # if relative:
  #  true = rescale(true, truncate=False)
  for k in range(true.shape[1]):
    if(False):
      print('Min True: ',np.min(true[0,k,0]))
      print('Max True: ',np.max(true[0,k,0]))
    plt.imshow(true[0,k,0], cmap='gray')
    plt.axis('off')
    plt.savefig(outputDir + 'true' '_k' + str(k) + '.png', bbox_inches='tight', pad_inches=0)
    

def save_all(args,relativeScale=False):
  '''
  Save all the pictures determining a reconstruction process
  '''

  # Store Image Prefix
  prefstr = args.imgprefix + '_'  

  # Save Original Image
  if(args.savetrue):
    trueFileName = args.maindir+'imgs_n1.npy'    
    if(os.path.exists(trueFileName)):
      if(args.printlevel>0):
        print('Saving true image: ',trueFileName)
      save_true(trueFileName,args.outputdir,relative=relativeScale)
  
  # Read Velocity Encoding if file exists
  vencfile = args.maindir+'venc_n1.npy'
  if(os.path.exists(vencfile)):
    venc = np.load(vencfile)
  else:
    venc = None

  # Loop over undersamplingn ratios
  for p in [0.25, 0.5, 0.75, 0.80, 0.85, 0.90, 0.95]:
    for samptype in ['bernoulli', 'vardengauss']:
      if(args.savemask):
        maskFile = args.maskdir + 'undersamplpattern_p' + str(int(p*100)) + samptype + '_n' + str(args.numsamples) + '.npy'
        if(os.path.exists(maskFile)):
          maskoutFile = args.outputdir + prefstr + 'undersamplpattern_p' + str(int(p*100)) + samptype + '_n' + str(args.numsamples) + '.png'
          if(args.printlevel > 0):
            print('Saving undersampling mask: ',maskFile)
          save_mask(maskFile, maskoutFile)           

      for noise_percent in [0.0, 0.01, 0.05, 0.1, 0.3]:
        for wavetype in ['HAAR','DB8']:
          # Loop on the reconstruction method
          for algtype in ['CS','CSDEB','OMP']:
            if(args.saverec):
              recnpy = args.recdir + 'rec_noise' + str(int(noise_percent*100)) + \
                                            '_p' + str(int(p*100)) + samptype + \
                                            '_n' + str(args.numsamples) + \
                                            '_w' + str(wavetype) + \
                                            '_a' + str(algtype) + '.npy'
              if(os.path.exists(recnpy)):
                if(args.printlevel > 0):
                  print('Saving image reconstructions: ',recnpy)
                save_rec(recnpy, venc, p, samptype, noise_percent, wavetype, algtype, prefstr, args.maindir, args.outputdir, args.limits, args.fluidmaskfile, singleChannel=args.singlechannel,relative=relativeScale)
                if(args.printlevel > 0):
                  print('Saving image reconstruction errors')
                save_rec_noise(recnpy, trueFileName, venc, p, samptype, noise_percent, wavetype, algtype, prefstr, args.outputdir, use_truth=args.usetrueasref, singleChannel=args.singlechannel)

          # Save linear reconstructions if appropriate
          if(os.path.exists(recnpy)):
            if(args.savelin):
              if(args.printlevel > 0):
                print('Saving linear reconstruction')
              save_lin(recnpy, venc, p, samptype, noise_percent, wavetype, args.numsamples, prefstr, args.maindir, args.outputdir, args.limits, args.fluidmaskfile, args.usemultipatterns, singleChannel=args.singlechannel,relative=relativeScale)

# MAIN 
if __name__ == '__main__':

  # Init parser
  parser = argparse.ArgumentParser(description='Generate result images.')

  # numRealizations
  parser.add_argument('-n', '--numsamples',
                      action=None,
                      # nargs='+',
                      const=None,
                      default=100,
                      type=int,
                      choices=None,
                      required=False,
                      help='number of repetitions',
                      metavar='',
                      dest='numsamples')

  # maindir
  parser.add_argument('-m', '--maindir',
                      action=None,
                      # nargs='+',
                      const=None,
                      default='./',
                      type=str,
                      choices=None,
                      required=False,
                      help='folder with original image and velocity encoding file',
                      metavar='',
                      dest='maindir')

  # recdir
  parser.add_argument('-r', '--recdir',
                      action=None,
                      # nargs='+',
                      const=None,
                      default='./',
                      type=str,
                      choices=None,
                      required=False,
                      help='folder with the reconstructed images',
                      metavar='',
                      dest='recdir')

  # maskdir
  parser.add_argument('-s', '--maskdir',
                      action=None,
                      # nargs='+',
                      const=None,
                      default='./',
                      type=str,
                      choices=None,
                      required=False,
                      help='folder containing the undesampling masks',
                      metavar='',
                      dest='maskdir')

  # outputdir
  parser.add_argument('-o', '--outputdir',
                      action=None,
                      # nargs='+',
                      const=None,
                      default='./',
                      type=str,
                      choices=None,
                      required=False,
                      help='image output folder',
                      metavar='',
                      dest='outputdir')

  # output image prefix
  parser.add_argument('--imgprefix',
                      action=None,
                      # nargs='+',
                      const=None,
                      default='',
                      type=str,
                      choices=None,
                      required=False,
                      help='output image prefix',
                      metavar='',
                      dest='imgprefix')

  # Use the True Images as a refences when evaluating noise
  parser.add_argument('--usetrueasref',
                      action='store_true',
                      default=False,
                      required=False,
                      help='Use original image instead of average image to evaluate noise',
                      dest='usetrueasref')

  # save_true
  parser.add_argument('--savetrue',
                      action='store_true',
                      default=False,
                      required=False,
                      help='save original image',
                      dest='savetrue')

  # save_mask
  parser.add_argument('--savemask',
                      action='store_true',
                      default=False,
                      required=False,
                      help='save undersampling mask',
                      dest='savemask')

  # save_rec
  parser.add_argument('--saverec',
                      action='store_true',
                      default=False,
                      required=False,
                      help='save reconstructed images',
                      dest='saverec')

  # save_noise
  parser.add_argument('--savenoise',
                      action='store_true',
                      default=False,
                      required=False,
                      help='save noise image',
                      dest='savenoise')

# save linear reconstructions
  parser.add_argument('--savelin',
                      action='store_true',
                      default=False,
                      required=False,
                      help='save linear reconstructions',
                      dest='savelin')

  # singlechannel
  parser.add_argument('--singlechannel',
                      action='store_true',
                      default=False,
                      required=False,
                      help='treat the image as single-channel, without velocity components',
                      dest='singlechannel')

  # Print Level
  parser.add_argument('-p', '--printlevel',
                      action=None,
                      # nargs='+',
                      const=None,
                      default=0,
                      type=float,
                      choices=None,
                      required=False,
                      help='print level, 0 - no print, >0 increasingly more information ',
                      metavar='',
                      dest='printlevel')

  # fluidmaskfile
  parser.add_argument('--fluidmaskfile',
                      action=None,
                      # nargs='+',
                      const=None,
                      default='',
                      type=str,
                      choices=None,
                      required=False,
                      help='name of the npy file containing the binary mask',
                      metavar='',
                      dest='fluidmaskfile')   

  # use unique undesampling pattern for each reconstruction (false) or change every time
  parser.add_argument('-um', '--usemultipatterns',
                      action='store_true',
                      default=False,
                      required=False,
                      help='generate a unique undersampling pattern for each noise realization',
                      dest='usemultipatterns')

  # limits
  parser.add_argument('--limits',
                      action=None,
                      nargs='*',
                      const=None,
                      default=[None,None,None,None,None,None,None,None],
                      type=float,
                      choices=None,
                      required=False,
                      help='list of images limits',
                      metavar='',
                      dest='limits')


  # Parse Commandline Arguments
  args = parser.parse_args()

  # Save all the images
  save_all(args)

  if(args.printlevel > 0):
    print('Completed!!!')  

