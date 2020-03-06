import sys
import os
# sys.path.append('../../')
sys.path.append('/home/dschiava/Documents/01_Development/01_PythonApps/03_Reconstruction/')
from recover import linear_reconstruction, recover_vel
from genSamples import getKspace
import numpy as np
import numpy.fft as fft
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, ScalarFormatter
from CSRecoverySuite import CSRecovery,CSRecoveryDebiasing, Operator4dFlow, pywt2array, array2pywt, crop
import argparse

home = os.getenv('HOME')

fs = 12
plt.rc('font',  family='serif')
plt.rc('xtick', labelsize='x-small')
plt.rc('ytick', labelsize='x-small')
plt.rc('text',  usetex=True)

def get_files(dir, noise, uval, utype, method, numsamples):
  fourier_file     = dir + 'noisy_noise' + str(int(noise*100)) + '_n' + str(numsamples) + '.npy'
  undersample_file = dir + 'undersamplpattern_p' + str(int(uval*100)) + utype + '_n' + str(numsamples) + '.npy'
  orig_file        = dir + 'imgs_n1.npy'
  recovered_file   = dir + method + '/' + 'rec_noise' + str(int(noise*100)) + '_p' + str(int(uval*100)) + utype + '_n' + str(numsamples) + '.npy'
  return fourier_file, orig_file, undersample_file, recovered_file

def get_complex(dir, noise, uval, utype, method, numsamples, usecompleximgs, usetrueimg, addlinearrec):
    
  # Get all files
  fourier_file, orig_file, undersample_file, recovered_file = get_files(dir, noise, uval, utype, method, numsamples)

  # Load reconstruction
  if(os.path.exists(recovered_file)):
    recovered = np.load(recovered_file)
  else:
    print('ERROR: file with reconstructions not found: ',recovered_file)
    sys.exit(-1)

  # Load original image
  if(os.path.exists(orig_file)):
    orig = np.load(orig_file)

  # Compute linear reconstructions if requested
  if(addlinearrec):
    
    # Load undesampling mask
    if(os.path.exists(undersample_file)):
      omega = np.load(undersample_file)
      if(len(omega.shape) == 3):
        omega = pattern[0]
    else:    
      print('ERROR: file with undersampling mask not found: ',undersample_file)
    
    # Compute linear reconstruction
    if(os.path.exists(fourier_file)):
      linrec = linear_reconstruction(fourier_file, omega)
    else:
      linrec = None
      print('WARNING: file with k-space image not found: ',fourier_file)

    return recovered, orig, linrec
  else:
    return recovered, orig, None

def get_final(dir, noise, uval, utype, method, numsamples, addlinearrec):

  # Get all files
  fourier_file, orig_file, undersample_file, recovered_file = get_files(dir, noise, uval, utype, method, numsamples)
  
  # Get Original image
  orig = np.load(orig_file) 
  new_shape = crop(orig[0,0,0]).shape
  linrec = linear_reconstruction(fourier_file, omega)
  
  # Get velocity encoding
  vencfile = dir + 'venc_n1.npy'
  if os.path.exists(vencfile):
      venc = np.load(dir + 'venc_n1' + '.npy')
  else:
      venc = None

  # Recover the velocity components
  imgs   = recover_vel(linrec, venc)
  csimgs = recover_vel(recovered, venc)

  # Return
  return csimgs, imgs, orig[0,:,:,:new_shape[0], :new_shape[1]]

def get_error(dir, noise, uval, utype, method, numsamples, usecompleximgs, usetrueimg, addlinearrec):
    
  # Get CS reconstructions, linear reconstruction and original image
  if usecompleximgs:
    csimgs, orig, linimgs = get_complex(dir, noise, uval, utype, method, numsamples, usecompleximgs, usetrueimg, addlinearrec)
  else:
    csimgs, orig, linimgs = get_final(dir, noise, uval, utype, method, numsamples, usecompleximgs, usetrueimg, addlinearrec) 

  # Get average cs and linear reconstructed image
  avgcs = csimgs.mean(axis=0)

  # Init vectors for MSE: one for every sample
  msecs = np.zeros(csimgs.shape[0])
  print('Max of original', np.amax(np.abs(orig)))
  print('Max of avg cs reconstruction: ', np.amax(np.abs(avgcs)))

  if(addlinearrec):
    avglin = linimgs.mean(axis=0)
    mselin = np.zeros(csimgs.shape[0])
    print('Max of avg linear reconstruction: ', np.amax(np.abs(avglin)))

  for k in range(csimgs.shape[0]):
    if usetrueimg:
      # Compare against the truth
      msecs[k]  = np.abs(((orig-csimgs[k])**2).mean())
      # Normalize with respect to the mean intensity of the original image
      msecs[k]  = msecs[k]/(np.mean(np.abs(orig))**2)

      if(addlinearrec):
        mselin[k] = np.abs(((orig-linimgs[k])**2).mean())
        mselin[k] = mselin[k]/(np.mean(np.abs(orig))**2)
    else: 
      # Compare against the average reconstructed image
      msecs[k]  = np.abs(((avgcs-csimgs[k])**2).mean())
      # Normalize with respect to the mean intensity of the average reconstruction
      msecs[k]  = msecs[k]/(np.mean(np.abs(avgcs))**2)

      if(addlinearrec):
        mselin[k] = np.abs(((avglin-linimgs[k])**2).mean())
        mselin[k] = mselin[k]/(np.mean(np.abs(avglin))**2)
  
  if(addlinearrec):
    return msecs, mselin
  else:
    return msecs, None

def get_folder(use_complex):    
    if use_complex:
        folder = '/plots/msecomplex'
    else:
        folder = '/plots/msefinal'
    return folder

def formatting(ax, lgd, xdesc):
    plt.ylabel('MSE',fontsize=fs)
    plt.tick_params(labelsize=fs)
    ax.set_xticks(range(1, len(lgd)+1))
    ax.set_xticklabels(lgd)
    plt.xlabel(xdesc)
    plt.tight_layout()

def plot_pdiff(dir, recdir, patterndir, noise_percent, p, type, num_samples, use_complex, use_truth, useCS):
    folder = get_folder(use_complex)
    fig, ax = plt.subplots(figsize=(4,3))
    i = 0
    colors = ['blue','orange','green', 'red']
    alpha = 1
    allplt = [None]*4
    count = 0
    for noise_percent in [0.01, 0.05, 0.1, 0.3]:#
        msecs, mselin = get_error(dir, recdir, patterndir, noise_percent, p, type, method, num_samples, use_complex, use_truth)
        if useCS:
            toplot = msecs
            msg = 'vplt'
        else:
            toplot = mselin
            msg = 'vplt_lin'
        if not use_truth:
            msg = msg + 'avg'
        alpha = alpha - 0.25
        allplt[i] = toplot
        i = i + 1
    bplts = plt.violinplot(allplt)
    for patch, color in zip(bplts['bodies'], colors):
        patch.set_facecolor(color)
    formatting(ax, ['1\%', '5\%', '10\%', '30\%'], 'Noise')
    if not os.path.exists(recdir + folder):
        os.makedirs(recdir+folder)
    fname = recdir + folder + '/' + msg + '_p' + str(int(p*100)) + type + '.pdf'
    plt.savefig(fname)
    print("Saved as " + fname)
    plt.close(fig)

def plot_noisediff(args):
  '''
  Ploting MSE for variable k-space noise
  '''
  colors = ['blue','orange','green', 'red', 'black']
  fig, ax = plt.subplots(figsize=(4,3))
  i = 0
  alpha = 1

  # Set the baseline conditions
  # These are the first element of the lists
  bl_noise  = args.noise[0]
  bl_uval   = args.uval[0]
  bl_utype  = args.utype[0]
  bl_method = args.method[0]

  if(args.singlechannel):
    numChannels = 1
  else:
    numChannels = 3

  allplt = [None]*len(args.noise)
  for noise in sorted(args.noise): 
    msecs, mselin = get_error(args.dir, noise, bl_uval, bl_utype, bl_method, args.numsamples, args.usecompleximgs, args.usetrueimg, args.addlinearrec)
    
    if(args.addlinearrec):
      toplot = mselin
      msg = 'vplt_lin'      
    else:
      toplot = msecs
      msg = 'vplt'

    if(not(args.usetrueimg)):
      msg = msg + 'avg'

    i = i + 1
    allplt[i-1] = toplot
    #allplt[int(p/0.25)-1] = toplot

  bplts = plt.violinplot(allplt)
  for patch, color in zip(bplts['bodies'], colors):
      patch.set_facecolor(color)
  
  plt.ylabel('MSE',fontsize=fs)
  plt.tick_params(labelsize=fs)
  lgd = [str(int(x*100)) + '\%' for x in sorted(args.noise)]
  ax.set_xticks(range(1, len(lgd)+1))
  ax.set_xticklabels(lgd)
  plt.xlabel('Undersampling')
  plt.tight_layout()

  # Save Img  
  fname = args.outputdir + msg + '_noise' + str(int(noise*100)) + bl_utype + '.pdf'
  plt.savefig(fname)
  print("MSE Image saved: " + fname)
  plt.close(fig)

def plot_methoddiff(dir, recdir, patterndir, noise_percent, p, type, num_samples, use_complex, use_truth):
    folder = get_folder(use_complex)
    colors = ['blue','orange','green', 'red']
    fig, ax = plt.subplots(figsize=(4,3))
    i = 0
    alpha = 1
    allplt = [None]*3
    for methodfolder in [solver_folder(0), solver_folder(1), solver_folder(2)]: 
        msecs, mselin = get_error(dir, recdir+methodfolder, patterndir, noise_percent, p, type, num_samples, use_complex, use_truth)
        #linear reconstruction case doesn't make sense for comparing methods
        toplot = msecs
        msg = 'vplt'
        if not use_truth:
            msg = msg + 'avg'
        allplt[i] = toplot
        i = i + 1
    bplts = plt.violinplot(allplt)
    for patch, color in zip(bplts['bodies'], colors):
        patch.set_facecolor(color)
    formatting(ax, ['CS', 'CSDEBIAS', 'OMP'], 'Solver')
    if not os.path.exists(recdir + folder):
        os.makedirs(recdir+folder)
    fname = recdir + folder + '/' + msg + '_noise' + str(int(noise_percent*100)) + '_p'+str(int(p*100)) + type + '.pdf'
    plt.savefig(fname)
    print("Saved as " + fname)
    plt.close(fig)

def pltviolin(dir, recdir, patterndir, num_samples, use_complex, use_truth):
    #use_complex: compare against complex images or final recovered velocity images
    #use_truth: compare against true values or the average recovered images 
    for p in [0.25]:
        for type in ['vardengauss']:#'bernoulli', 'bpoisson']:
            for noise in [0.01, 0.05, 0.1, 0.3]:
                try: #plot comparison of undersampling % for CS and linear rec. images
                    plotnoisediff(dir, recdir, patterndir,  noise, p, type, num_samples, use_complex, use_truth, True)
                    plotnoisediff(dir, recdir, patterndir,  noise, p, type, num_samples, use_complex, use_truth, False)
                except Exception as e:
                    print(e)
                    print('Not found: recovered images with ', noise, 'noise', p, 'p', type, 'type')
                    continue
    for p in [0.25, 0.5, 0.75]:
        for type in ['vardengauss']:#'bernoulli', 'bpoisson']:
            for noise in [0.01]:
                try:
                    plotpdiff(dir, recdir, patterndir,  noise, p, type, num_samples, use_complex, use_truth, True)
                    plotpdiff(dir, recdir, patterndir,  noise, p, type, num_samples, use_complex, use_truth, False)
                except Exception as e:
                    print(e)
                    print("Not found: recovered images with ", noise, 'noise', p, 'p', type, 'type')
                    continue
    
# MAIN 
if __name__ == '__main__':

  # Init parser
  parser = argparse.ArgumentParser(description='Generate result images.')

  # Load Base Line Params
  # noise
  parser.add_argument('-n', '--noise',
                      action=None,
                      nargs='*',
                      const=None,
                      default=[0.1],
                      type=float,
                      choices=None,
                      required=False,
                      help='list of noise values',
                      metavar='',
                      dest='noise')
  # uval
  parser.add_argument('-u', '--uval',
                      action=None,
                      nargs='*',
                      const=None,
                      default=[0.75],
                      type=float,
                      choices=None,
                      required=False,
                      help='list of undersampling ratios',
                      metavar='',
                      dest='uval')
  # utype
  parser.add_argument('-t', '--utype',
                      action=None,
                      nargs='*',
                      const=None,
                      default=['vardengauss'],
                      type=str,
                      choices=None,
                      required=False,
                      help='list of random undersampling patterns',
                      metavar='',
                      dest='utype')
  # numsamples
  parser.add_argument('-s', '--numsamples',
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
  # numpts
  parser.add_argument('-np', '--numpts',
                      action=None,
                      # nargs='+',
                      const=None,
                      default=100,
                      type=int,
                      choices=None,
                      required=False,
                      help='number of points for computing the',
                      metavar='',
                      dest='numpts')
  # method
  parser.add_argument('-m', '--method',
                      action=None,
                      nargs='*',
                      const=None,
                      default=['cs'],
                      type=str,
                      choices=None,
                      required=False,
                      help='list of reconstruction methods',
                      metavar='',
                      dest='method')
  # maindir
  parser.add_argument('-d', '--dir',
                      action=None,
                      # nargs='+',
                      const=None,
                      default='./',
                      type=str,
                      choices=None,
                      required=False,
                      help='folder containing the file with the point locations for the correlation and with subfoldes with the reconstruction methods',
                      metavar='',
                      dest='dir')
  # outputdir
  parser.add_argument('-o', '--outputdir',
                      action=None,
                      # nargs='+',
                      const=None,
                      default='./',
                      type=str,
                      choices=None,
                      required=False,
                      help='output folder for images',
                      metavar='',
                      dest='outputdir')
  # singlechannel
  parser.add_argument('--singlechannel',
                      action='store_true',
                      default=False,
                      required=False,
                      help='treat the image as single-channel, without velocity components',
                      dest='singlechannel')    
  # singlechannel
  parser.add_argument('--usecompleximgs',
                      action='store_true',
                      default=False,
                      required=False,
                      help='evaluate the MSE based on the reconstructed complex image. Default: use velocity components.',
                      dest='usecompleximgs')    
  # addlinear reconstructions
  parser.add_argument('--addlinearrec',
                      action='store_true',
                      default=False,
                      required=False,
                      help='Add linear reconstruction MSE to the plots.',
                      dest='addlinearrec')      
  # singlechannel
  parser.add_argument('--usetrueimg',
                      action='store_true',
                      default=False,
                      required=False,
                      help='use the true image to evaluate the MSE',
                      dest='usetrueimg')    
  # Print Level
  parser.add_argument('-p', '--printlevel',
                      action=None,
                      # nargs='+',
                      const=None,
                      default=0,
                      type=int,
                      choices=None,
                      required=False,
                      help='print level, 0 - no print, >0 increasingly more information ',
                      metavar='',
                      dest='printlevel')

  # Parse Commandline Arguments
  args = parser.parse_args()

  # Plot the parameter perturbations
  if(len(args.noise) > 1):
    plot_noisediff(args)
  #if(len(args.uval) > 1):
  #  plot_pdiff(args)
  #if(len(args.utype) > 1):
  #  plot_sampdiff(args)
  #if(len(args.method) > 1):
  #  plot_methoddiff(args)

  # Completed!
  if(args.printlevel > 0):
    print('Completed!!!')  



    









