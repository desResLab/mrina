import sys,os,math
import numpy as np
import matplotlib.pyplot as plt
sys.path.append('../')
from mrina.mri_utils import get_umask_string, get_method_string, get_wavelet_string
import argparse

home = os.getenv('HOME')

fs = 12
plt.rc('font',  family='serif')
plt.rc('xtick', labelsize='x-small')
plt.rc('ytick', labelsize='x-small')
plt.rc('text',  usetex=True)
start = 0
end   = 15 # last distance to include in plot (final x axis value)
interval = int((end-start)/4) #int(math.ceil(((end-start)/4) / 10.0)) * 10))

def getCorrelationFileName(numsamples, numpts, noise, p, masktype, wavelet, method, useFluidMask):
  res = 'corrcoeff' + str(numpts) + \
        '_noise' + str(int(noise*100)) + \
        '_p' + str(int(p*100)) + masktype + \
        '_n'+str(numsamples)+ \
        '_w' + str(get_wavelet_string(wavelet)) + \
        '_a' + str(get_method_string(method))
  if(useFluidMask):
    res += '_fluid.npy'
  else:
    res += '.npy'
  return res

def get_coeff(noise, p, masktype, wavelet, method, numsamples, numpts, dir, useFluidMask):
  '''
  Retrieve the file with the correlations
  ''' 
  corrfile = dir + get_method_string(method) + '/' + getCorrelationFileName(numsamples, numpts, noise, p, masktype, wavelet, method, useFluidMask)
  if(os.path.isfile(corrfile)):
    coeff = np.load(corrfile)
    return coeff
  else:
    print('Warning: no correlation file found: ',corrfile)
    return None

def plot_corr(noise_percent, p, samptype, wavelet, method, n, num_pts, v, dir, labelstr, useFluidMask):
  coeff = get_coeff(noise_percent, p, samptype, wavelet, method, n, num_pts, dir, useFluidMask)
  if(coeff is None):
    return None
  else:
    coeff = coeff[v]
    coeff[np.isnan(coeff)] = 0
    corravg = np.mean(coeff, axis=1)
    corrmin = np.percentile(coeff, 10, axis=1)
    corrmax = np.percentile(coeff, 90, axis=1)
    size = len(corravg)
    p, = plt.plot(range(start+1,end+1), corravg[start:end], label=labelstr)
    plt.fill_between(range(start+1,end+1), corrmin[start:end], corrmax[start:end], alpha=0.2) # label='10-90 CI')
    return p

def plot_noisediff(args, save_fig=True):
  
  print("Plotting correlations for various k-space noise values...")

  # Set the baseline conditions
  # These are the first element of the lists
  bl_noise   = args.noise[0]
  bl_uval    = args.uval[0]
  bl_utype   = args.utype[0]
  bl_method  = args.method[0]
  bl_wavelet = args.wavelet[0]

  if(args.singlechannel):
    numChannels = 1
  else:
    numChannels = 4
  
  # Loop on the reconstruction component
  for k in range(0,numChannels):

    plt.figure(figsize=(2.2,4))
    # Loop on all noise values
    for noise in args.noise:
      label = r"{}\% noise".format(int(noise*100))
      plot_corr(noise, bl_uval, bl_utype, bl_wavelet, bl_method, args.numsamples, args.numpts, k, args.dir, label, args.usefluidmask)
    
    plt.xlabel('Distance [px]',fontsize=fs)
    plt.ylabel('Correlation Coefficient',fontsize=fs)
    # plt.axhline(y=0,c='gray',lw=1,ls='--',alpha=0.8)
    plt.grid(color='gray', linestyle='-', linewidth=0.5, alpha=0.2)
    plt.tick_params(labelsize=fs)
    plt.ylim([-0.2,1.0])
    plt.xlim([1.0,9.0])
    plt.xticks(np.arange(1, 11, 2))
    plt.legend(loc='upper center',fontsize=fs-2)
    plt.tight_layout()

    if save_fig:
      plt.savefig(args.outputdir + 'diffnoise' + str(start) + 'to' + str(end) + '_p' + str(int(bl_uval*100)) + bl_utype + '_k' + str(k) + '.pdf')
      plt.close()
    else:
      plt.show()

def plot_pdiff(args, save_fig=True):
  
  print("Plotting correlations for various undersampling ratios...")

  # Set the baseline conditions
  # These are the first element of the lists
  bl_noise   = args.noise[0]
  bl_uval    = args.uval[0]
  bl_utype   = args.utype[0]
  bl_method  = args.method[0]
  bl_wavelet = args.wavelet[0]

  if(args.singlechannel):
    numChannels = 1
  else:
    numChannels = 4

  # Loop on the reconstruction component
  for k in range(0,numChannels):

    plt.figure(figsize=(2.2,4))
    # Loop on all noise values
    for p in args.uval:
      label = r"{}\%".format(int(p*100))
      plot_corr(bl_noise, p, bl_utype, bl_wavelet, bl_method, args.numsamples, args.numpts, k, args.dir, label, args.usefluidmask)
    
    plt.xlabel('Distance [px]',fontsize=fs)
    plt.ylabel('Correlation Coefficient',fontsize=fs)
    # plt.axhline(y=0,c='gray',lw=1,ls='--',alpha=0.8)
    plt.grid(color='gray', linestyle='-', linewidth=0.5, alpha=0.2)
    plt.tick_params(labelsize=fs)
    plt.ylim([-0.2,1.0])
    plt.xlim([1.0,9.0])
    plt.xticks(np.arange(1, 11, 2))
    plt.legend(loc='upper center',fontsize=fs-2)
    plt.tight_layout()

    if save_fig:
      plt.savefig(args.outputdir + 'diffuratio' + str(start) + 'to' + str(end) + '_p' + str(int(bl_uval*100)) + bl_utype + '_k' + str(k) + '.pdf')
      plt.close()
    else:
      plt.show()

def plot_sampdiff(args, save_fig=True):
    
  print("Plotting correlations for various undersampling patterns...")

  # Set the baseline conditions
  # These are the first element of the lists
  bl_noise   = args.noise[0]
  bl_uval    = args.uval[0]
  bl_utype   = args.utype[0]
  bl_method  = args.method[0]
  bl_wavelet = args.wavelet[0]

  if(args.singlechannel):
    numChannels = 1
  else:
    numChannels = 4

  # Loop on the reconstruction component
  for k in range(0,numChannels):

    plt.figure(figsize=(2.2,4))
    # Loop on all noise values
    for samptype in args.utype:
      label = get_umask_string(samptype)
      plot_corr(bl_noise, bl_uval, samptype, bl_wavelet, bl_method, args.numsamples, args.numpts, k, args.dir, label, args.usefluidmask)
    
    plt.xlabel('Distance [px]',fontsize=fs)
    plt.ylabel('Correlation Coefficient',fontsize=fs)
    # plt.axhline(y=0,c='gray',lw=1,ls='--',alpha=0.8)
    plt.grid(color='gray', linestyle='-', linewidth=0.5, alpha=0.2)
    plt.tick_params(labelsize=fs)
    plt.ylim([-0.2,1.0])
    plt.xlim([1.0,9.0])
    plt.xticks(np.arange(1, 11, 2))
    plt.legend(loc='upper center',fontsize=fs-2)
    plt.tight_layout()

    if save_fig:
      plt.savefig(args.outputdir + 'diffumask' + str(start) + 'to' + str(end) + '_p' + str(int(bl_uval*100)) + bl_utype + '_k' + str(k) + '.pdf')
      plt.close()
    else:
      plt.show()            

def plot_methoddiff(args, save_fig=True):

  print("Plotting correlations for various reconstruction algorithms...")

  # Set the baseline conditions
  # These are the first element of the lists
  bl_noise   = args.noise[0]
  bl_uval    = args.uval[0]
  bl_utype   = args.utype[0]
  bl_method  = args.method[0]
  bl_wavelet = args.wavelet[0]

  if(args.singlechannel):
    numChannels = 1
  else:
    numChannels = 4

  # Loop on the reconstruction component
  for k in range(0,numChannels):

    plt.figure(figsize=(2.2,4))
    # Loop on all noise values
    for method in args.method:
      label = get_method_string(method)
      plot_corr(bl_noise, bl_uval, bl_utype, bl_wavelet, method, args.numsamples, args.numpts, k, args.dir, label, args.usefluidmask)
    
    plt.xlabel('Distance [px]',fontsize=fs)
    plt.ylabel('Correlation Coefficient',fontsize=fs)
    # plt.axhline(y=0,c='gray',lw=1,ls='--',alpha=0.8)
    plt.grid(color='gray', linestyle='-', linewidth=0.5, alpha=0.2)
    plt.tick_params(labelsize=fs)
    plt.ylim([-0.2,1.0])
    plt.xlim([1.0,9.0])
    plt.xticks(np.arange(1, 11, 2))
    plt.legend(loc='upper center',fontsize=fs-2)
    plt.tight_layout()

    if save_fig:
      plt.savefig(args.outputdir + 'diffmethod' + str(start) + 'to' + str(end) + '_p' + str(int(bl_uval*100)) + bl_utype + '_k' + str(k) + '.pdf')
      plt.close()
    else:
      plt.show()   

def plot_wavediff(args, save_fig=True):

  print("Plotting correlations for various wavelet basis...")

  # Set the baseline conditions
  # These are the first element of the lists
  bl_noise   = args.noise[0]
  bl_uval    = args.uval[0]
  bl_utype   = args.utype[0]
  bl_method  = args.method[0]
  bl_wavelet = args.wavelet[0]

  if(args.singlechannel):
    numChannels = 1
  else:
    numChannels = 4

  # Loop on the reconstruction component
  for k in range(0,numChannels):

    plt.figure(figsize=(2.2,4))
    # Loop on all noise values
    for wavelet in args.wavelet:
      label = get_wavelet_string(wavelet)
      plot_corr(bl_noise, bl_uval, bl_utype, wavelet, bl_method, args.numsamples, args.numpts, k, args.dir, label, args.usefluidmask)
    
    plt.xlabel('Distance [px]',fontsize=fs)
    plt.ylabel('Correlation Coefficient',fontsize=fs)
    plt.grid(color='gray', linestyle='-', linewidth=0.5, alpha=0.2)
    plt.tick_params(labelsize=fs)
    plt.ylim([-0.2,1.0])
    plt.xlim([1.0,9.0])
    plt.xticks(np.arange(1, 11, 2))
    plt.legend(loc='upper center',fontsize=fs-2)
    plt.tight_layout()

    if save_fig:
      plt.savefig(args.outputdir + 'diffwave' + str(start) + 'to' + str(end) + '_p' + str(int(bl_uval*100)) + bl_utype + '_k' + str(k) + '.pdf')
      plt.close()
    else:
      plt.show()   

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
  parser.add_argument('-w', '--wavelet',
                      action=None,
                      nargs='*',
                      const=None,
                      default=['haar'],
                      type=str,
                      choices=['haar','db8'],
                      required=False,
                      help='list of wavelet basis',
                      metavar='',
                      dest='wavelet')

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

  # usefluidmask
  parser.add_argument('--usefluidmask',
                      action='store_true',
                      default=False,
                      required=False,
                      help='compute correlation only within fluid region',
                      dest='usefluidmask')  

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
  if(len(args.uval) > 1):
    plot_pdiff(args)
  if(len(args.utype) > 1):
    plot_sampdiff(args)
  if(len(args.method) > 1):
    plot_methoddiff(args)
  if(len(args.method) > 1):
    plot_wavediff(args)

  # Completed!
  if(args.printlevel > 0):
    print('Completed!!!')  
