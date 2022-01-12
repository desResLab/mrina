import sys
import os
import warnings
sys.path.append('../')
from mrina.recover import linear_reconstruction, recover_vel
import numpy as np
import matplotlib.pyplot as plt
from mrina.mri_utils import isvalidcrop, extractFluidMask
from mrina.mri_utils import get_umask_string, get_method_string, get_wavelet_string
import argparse

warnings.filterwarnings('error')

home = os.getenv('HOME')
hatchPatterns = ['-', '+', 'x', '\\', '*', 'o', 'O', '.']

fs = 12
plt.rc('font',  family='serif')
plt.rc('xtick', labelsize='x-small')
plt.rc('ytick', labelsize='x-small')
plt.rc('text',  usetex=True)

def generateViolinPlot(lgd,xlabel,allplt,allplt_lin=None):
  fig, ax = plt.subplots(figsize=(3,4))

  bplts = plt.violinplot(allplt,showmeans=False, showmedians=False)
  for patch in bplts['bodies']:
      patch.set_facecolor('r')
      patch.set_edgecolor('r')
      patch.set_linewidth(0.8)
      patch.set_alpha(0.5)

  if(not(allplt_lin is None)):
    bplts = plt.violinplot(allplt_lin,showmeans=False, showmedians=False)
    for patch in bplts['bodies']:
      patch.set_facecolor('gray')
      patch.set_edgecolor('gray')
      patch.set_linewidth(0.8)
      patch.set_alpha(0.5)
  
  plt.ylabel('MSE',fontsize=fs)
  plt.tick_params(labelsize=fs)
  ax.set_xticks(range(1, len(lgd)+1))
  ax.set_xticklabels(lgd)
  plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0), useOffset=True)
  plt.xlabel(xlabel)
  plt.tight_layout()

  return fig

def generateBarPlot(lgd,xlabel,ylabel,
                    usecompleximgs,addlinearrec,
                    allplt_mag,allplt_ang,allplt_cmx,
                    allplt_mag_lin=None,allplt_ang_lin=None,allplt_cmx_lin=None,
                    rotext=False):

  # Set Space and Initial Shift for the plots
  pltShift = 1.0
  pltNum = 4
  if(addlinearrec):
    if(usecompleximgs):
      pltShift = 2.5
      pltNum = 7 
    else:
      pltShift = 0.5
      pltNum = 2
  else:
    if(usecompleximgs):
      pltShift = 1.0
      pltNum = 4
    else:
      pltShift = 0.0
      pltNum = 2

  # New picture
  fig, ax = plt.subplots(figsize=(4,3))

  # Set Values on the X Axis
  xVals_mag = np.arange(0,pltNum*len(allplt_mag),pltNum)
  if(usecompleximgs):
    xVals_ang = np.arange(1,pltNum*len(allplt_ang),pltNum)
    xVals_cmx = np.arange(2,pltNum*len(allplt_cmx),pltNum)
  
  # Compute Mean values
  # Magnitude
  yVals_mag = np.zeros(len(allplt_mag))
  for loopA in range(len(allplt_mag)):
    yVals_mag[loopA] = allplt_mag[loopA].mean()
  
  # Angle
  yVals_ang = np.zeros(len(allplt_ang))
  for loopA in range(len(allplt_ang)):
    yVals_ang[loopA] = allplt_ang[loopA].mean()

  # Complex
  yVals_cmx = np.zeros(len(allplt_cmx))
  for loopA in range(len(allplt_cmx)):
    yVals_cmx[loopA] = allplt_cmx[loopA].mean()
  
  # Compute Errors
  # Magnitude
  yErr_mag = np.zeros((2,len(allplt_mag)))
  for loopA in range(len(allplt_mag)):
    yErr_mag[0,loopA] = np.abs(np.percentile(allplt_mag[loopA],10)-yVals_mag[loopA])
    yErr_mag[1,loopA] = np.abs(np.percentile(allplt_mag[loopA],90)-yVals_mag[loopA])
  # Angle
  yErr_ang = np.zeros((2,len(allplt_ang)))
  for loopA in range(len(allplt_ang)):
    yErr_ang[0,loopA] = np.abs(np.percentile(allplt_ang[loopA],10)-yVals_ang[loopA])
    yErr_ang[1,loopA] = np.abs(np.percentile(allplt_ang[loopA],90)-yVals_ang[loopA])
  # Complex
  yErr_cmx = np.zeros((2,len(allplt_cmx)))
  for loopA in range(len(allplt_cmx)):
    yErr_cmx[0,loopA] = np.abs(np.percentile(allplt_cmx[loopA],10)-yVals_cmx[loopA])
    yErr_cmx[1,loopA] = np.abs(np.percentile(allplt_cmx[loopA],90)-yVals_cmx[loopA])

  # Plot MSE
  ax.bar(xVals_mag, yVals_mag, yerr=yErr_mag, width=0.8, error_kw=dict(lw=0.5, capsize=2, capthick=0.5),label='cs mag',hatch='++')
  if(usecompleximgs):
    ax.bar(xVals_ang, yVals_ang, yerr=yErr_ang, width=0.8, error_kw=dict(lw=0.5, capsize=2, capthick=0.5),label='cs ang',hatch='--')
    ax.bar(xVals_cmx, yVals_cmx, yerr=yErr_cmx, width=0.8, error_kw=dict(lw=0.5, capsize=2, capthick=0.5),label='cs cmx',hatch='\\\\')

  # Plot MSE for linear reconstruction
  if(addlinearrec):
    
    # Set X values
    if(usecompleximgs):
      xVals_mag = np.arange(3,pltNum*len(allplt_mag_lin),pltNum)
      xVals_ang = np.arange(4,pltNum*len(allplt_ang_lin),pltNum)
      xVals_cmx = np.arange(5,pltNum*len(allplt_cmx_lin),pltNum)
    else:
      xVals_mag = np.arange(1,pltNum*len(allplt_mag_lin),pltNum)

    # Compute Mean values
    yVals_mag = np.zeros(len(allplt_mag_lin))
    for loopA in range(len(allplt_mag_lin)):
      yVals_mag[loopA] = allplt_mag_lin[loopA].mean()

    #
    yVals_ang = np.zeros(len(allplt_ang_lin))
    for loopA in range(len(allplt_ang_lin)):
      yVals_ang[loopA] = allplt_ang_lin[loopA].mean()

    #
    yVals_cmx = np.zeros(len(allplt_cmx_lin))
    for loopA in range(len(allplt_cmx_lin)):
      yVals_cmx[loopA] = allplt_cmx_lin[loopA].mean()

    # Compute Errors
    yErr_mag = np.zeros((2,len(allplt_mag_lin)))
    for loopA in range(len(allplt_mag_lin)):
      yErr_mag[0,loopA] = np.abs(np.percentile(allplt_mag_lin[loopA],10)-yVals_mag[loopA])
      yErr_mag[1,loopA] = np.abs(np.percentile(allplt_mag_lin[loopA],90)-yVals_mag[loopA])
    #
    yErr_ang = np.zeros((2,len(allplt_ang_lin)))
    for loopA in range(len(allplt_ang_lin)):
      yErr_ang[0,loopA] = np.abs(np.percentile(allplt_ang_lin[loopA],10)-yVals_ang[loopA])
      yErr_ang[1,loopA] = np.abs(np.percentile(allplt_ang_lin[loopA],90)-yVals_ang[loopA])
    #
    yErr_cmx = np.zeros((2,len(allplt_cmx_lin)))
    for loopA in range(len(allplt_cmx_lin)):
      yErr_cmx[0,loopA] = np.abs(np.percentile(allplt_cmx_lin[loopA],10)-yVals_cmx[loopA])
      yErr_cmx[1,loopA] = np.abs(np.percentile(allplt_cmx_lin[loopA],90)-yVals_cmx[loopA])

    ax.bar(xVals_mag, yVals_mag, yerr=yErr_mag, width=0.8, error_kw=dict(lw=0.5, capsize=2, capthick=0.5),label='lin mag',hatch='xx')
    if(usecompleximgs):
      ax.bar(xVals_ang, yVals_ang, yerr=yErr_ang, width=0.8, error_kw=dict(lw=0.5, capsize=2, capthick=0.5),label='lin ang',hatch='oo')
      ax.bar(xVals_cmx, yVals_cmx, yerr=yErr_cmx, width=0.8, error_kw=dict(lw=0.5, capsize=2, capthick=0.5),label='lin cmx',hatch='..')
   
  # Set X thicks
  ax.set_xticks(pltShift+pltNum*np.arange(len(lgd)))

  plt.ylabel(ylabel,fontsize=fs)
  if(rotext):
    ax.set_xticklabels(lgd, rotation=40, ha='right')  
  else:
    ax.set_xticklabels(lgd)
  plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0), useOffset=True)
  plt.legend(loc='lower center', fontsize=fs-4, bbox_to_anchor=(0.5, 1.0), 
             fancybox=False, shadow=False, ncol=3, handlelength=1.5, 
             labelspacing=0.2, columnspacing=5.25)
  plt.yscale('log')
  plt.tick_params(axis='both', reset=True, grid_color='gray', grid_alpha=0.1, 
                  which='both',bottom=True,left=True,top=False,right=False)
  plt.grid(True, axis='both', which='both')
  plt.rc('axes', axisbelow=True)
  plt.xlabel(xlabel)
  plt.tight_layout()

  return fig

def get_files(dir, maskdir, noise, uval, utype, wavelet, method, numsamples, usemultipatterns):
  # Get Image File Names
  orig_file        = dir + 'imgs_n1.npy'
  fourier_file     = dir + 'noisy_noise' + str(int(noise*100)) + '_n' + str(numsamples) + '.npy'
  if(usemultipatterns):
    mask_file = maskdir + 'undersamplpattern_p' + str(int(uval*100)) + utype + '_n' + str(numsamples) + '.npy'
  else:
    mask_file = maskdir + 'undersamplpattern_p' + str(int(uval*100)) + utype + '.npy'
  recovered_file   = dir + get_method_string(method) + '/' + 'rec_noise' + str(int(noise*100)) + \
                                          '_p' + str(int(uval*100)) + utype + \
                                          '_n' + str(numsamples) + \
                                          '_w' + str(get_wavelet_string(wavelet)) + \
                                          '_a' + str(get_method_string(method)) + '.npy'

  # print('orig_file',orig_file)
  # print('fourier_file',fourier_file)
  # print('mask_file',mask_file)
  # print('recovered_file',recovered_file)

  # Check if files exist and open the images
  # Original image
  if(os.path.exists(orig_file)):
    try:
      orig = np.load(orig_file).astype(np.complex)
    except Warning:
      print('ERROR: Cannot read original image - ',orig)
      sys.exit(-1)
    if(not(isvalidcrop(orig))):
      print('ERROR: Invalid original image file.')
      sys.exit(-1)
  else:
    print('ERROR: original image file not found: ',orig_file)
    sys.exit(-1)

  # Noisy Fourier Image
  if(os.path.exists(fourier_file)):
    try:
      fourier = np.load(fourier_file).astype(np.complex)
    except Warning:
      print('ERROR: Cannot read Fourier file - ',fourier_file)
      sys.exit(-1)
  else:
    print('ERROR: file for noisy Fourier image not found: ',fourier_file)
    sys.exit(-1)

  # Undersampling Mask
  if(os.path.exists(mask_file)):
    try:
      mask = np.load(mask_file).astype(bool)
    except Warning:
      print('ERROR: Cannot read binary mask - ',mask_file)
      sys.exit(-1)
  else:
    print('ERROR: undersampling mask file not found: ',mask_file)
    sys.exit(-1)

  # File with reconstructed image
  if(os.path.exists(recovered_file)):
    try:
      recovered = np.load(recovered_file).astype(np.complex)
    except Warning:
      print('ERROR: Cannot read reconstructedd image - ',recovered_file)
      sys.exit(-1)
  else:
    print('ERROR: file for reconstructed images not found: ',recovered_file)
    sys.exit(-1)

  return fourier, orig, mask, recovered

def get_complex(channel, dir, maskdir, noise, uval, utype, wavelet, method, numsamples, addlinearrec, usemultipatterns):
    
  # Get all files
  fourier, orig, mask, recovered = get_files(dir, maskdir, noise, uval, utype, wavelet, method, numsamples, usemultipatterns)

  # Get velocity encoding
  venc_file = dir + 'venc_n1.npy'
  if os.path.exists(venc_file):
    venc = np.load(dir + 'venc_n1.npy').astype(np.float)
  else:
    venc = None
    print('WARNING: file for velocity encoding not found: ',venc_file)  

  # Compute linear reconstructions if requested
  retLinRec = None
  if(addlinearrec):
    # Perform linear reconstruction
    linrec = linear_reconstruction(fourier,mask)
    # Return complex images only for the requested channel
    retLinRec = linrec[:,channel,:,:,:]

  # Convert True Image to Complex
  kspace_orig = np.zeros(orig.shape,dtype=complex)
  referencephase = np.zeros(orig[:,0].shape)
  magnitudes = np.ones(orig.shape)
  mag = orig[0,0]
  refphase = referencephase[0]
  vel = orig[0,1:4]
  compyk = np.zeros(vel.shape, dtype=complex)
  for k in range(orig.shape[2]):
      kspace_orig[0,0,k] = mag[k]*np.exp(1j*refphase[k])
      for j in range(0,orig.shape[1]-1):
          kspace_orig[0,j+1,k] = magnitudes[0,j+1,k]*np.exp(1j*refphase[k])*np.exp(np.pi*1j*vel[j,k]/venc)

  # Return complex images only for the requested channel
  return recovered[:,channel,:,:,:], kspace_orig[:,channel,:,:,:], retLinRec

def get_final(channel, dir, maskdir, noise, uval, utype, wavelet, method, numsamples, addlinearrec, usemultipatterns):

  # Get all files
  fourier, orig, mask, recovered = get_files(dir, maskdir, noise, uval, utype, wavelet, method, numsamples, usemultipatterns)

  # Get velocity encoding
  venc_file = dir + 'venc_n1.npy'
  if os.path.exists(venc_file):
    venc = np.load(dir + 'venc_n1.npy').astype(np.float)
  else:
    venc = None
    print('WARNING: file for velocity encoding not found: ',venc_file)

  # Recover the velocity components  
  csvels = recover_vel(recovered, venc)

  # Compute linear reconstructions if requested
  retLinVels = None
  if(addlinearrec):
    # Perform linear reconstruction
    linrec = linear_reconstruction(fourier,mask)
    # Compute velocities from linear reconstruction
    linvels = recover_vel(linrec, venc)
    retLinVels = linvels[:,channel,:,:,:]
    
  # Return
  return csvels[:,channel,:,:,:], orig[:,channel,:,:,:], retLinVels

def get_perc_diff(k,csimgs,refimg,useFluidMask=False,fluidMask=None):
  '''
  Evaluate the percentage 
  '''
  refimg[0,0][np.abs(refimg[0,0]) < 1.0e-3] = 1.0e-3
  res = np.absolute((np.absolute(refimg[0,0])-np.absolute(csimgs[k,0]))/(np.absolute(refimg[0,0])))
  if(useFluidMask):
    res = np.median(res[fluidMask])
  else:
    res = np.median(res)
  return res

def get_mse_mag(k,csimgs,refimg,useFluidMask=False,fluidMask=None):
  if(useFluidMask):
    imgNum = (np.absolute(refimg[0,0])-np.absolute(csimgs[k,0]))**2
    imgDen = np.absolute(refimg[0,0])**2
    res = np.mean(imgNum[fluidMask])
    den = np.mean(imgDen[fluidMask])
    if(den > 1.0e-12):
      res = res/den
  else:
    res = np.mean((np.absolute(refimg[0,0])-np.absolute(csimgs[k,0]))**2)    
    if(np.mean(np.absolute(refimg[0,0])**2) > 1.0e-12):
      res = res/np.mean(np.absolute(refimg[0,0])**2)    

  return res

def get_mse_ang(k,csimgs,refimg,useFluidMask=False,fluidMask=None):
  if(useFluidMask):
    imgNum = (np.angle(refimg[0,0])-np.angle(csimgs[k,0]))**2
    res = np.mean(imgNum[fluidMask])
  else:
    res = np.mean((np.angle(refimg[0,0])-np.angle(csimgs[k,0]))**2)   
    # mse_ang_cs[k]  = mse_ang_cs[k]/np.mean(np.angle(refimg)**2)

  return res

def get_mse_cmx(k,csimgs,refimg,useFluidMask=False,fluidMask=None):
  if(useFluidMask):
    imgNum = (np.absolute(refimg[0,0]-csimgs[k,0]))**2
    imgDen = np.absolute(refimg[0,0])**2
    res = np.mean(imgNum[fluidMask])
    den = np.mean(imgDen[fluidMask])

    if(den > 1.0e-12):
      res = res/den    
  else:

    res = np.mean((np.absolute(refimg[0,0]-csimgs[k,0]))**2)
    if(np.mean(np.absolute(refimg[0,0])**2) > 1.0e-12):
      res = res/np.mean(np.absolute(refimg[0,0])**2)

  return res

def get_error(channel, dir, maskdir, noise, uval, utype, wavelet, method, numsamples, usecompleximgs, usetrueimg, addlinearrec, useFluidMask, fluidMaskFile, usemultipatterns):
    
  # Get CS reconstructions, linear reconstruction and original image
  if usecompleximgs:
    csimgs, orig, linimgs = get_complex(channel, dir, maskdir, noise, uval, utype, wavelet, method, numsamples, addlinearrec, usemultipatterns)
  else:
    csimgs, orig, linimgs = get_final(channel, dir, maskdir, noise, uval, utype, wavelet, method, numsamples, addlinearrec, usemultipatterns) 

  # If use fluid mask, then compute the mask using the original image density+velocities
  fluidMask = None
  if(useFluidMask):

    if(fluidMaskFile == ''):
      orig_forMask = np.load(dir + 'imgs_n1.npy').astype(np.complex)
      fluidMask = extractFluidMask(orig_forMask)
      print('Using automatically extracted fluid mask...')
    else:
      fluidMask = np.load(dir + fluidMaskFile)
      print('Using fluid mask from file: '+ dir + fluidMaskFile)

  # Get average cs and linear reconstructed image
  if(usetrueimg):
    refimg = orig
  else:
    refimg = csimgs.mean(axis=0,keepdims=True)

  # DEBUG - CHECK THE DIFFERENCE IN RECONSTRUCTED IMAGE
  # plt.figure(figsize=(10,3))
  # print('noise ' + str(noise) + ' uval ' + str(uval) + ' utype ' + str(utype) + ' wave ' + str(wavelet) + ' alg ' + str(method))
  # if usecompleximgs:
  #   print('Using Complex Images')
  # else:
  #   print('Using Components')
  # plt.subplot(1,3,1)
  # plt.title('TRUE')
  # plt.imshow(np.absolute(refimg[0,0]))
  # plt.colorbar()
  # plt.subplot(1,3,2)
  # plt.title('REC')
  # plt.imshow(np.absolute(csimgs[0,0]))
  # plt.colorbar()
  # plt.subplot(1,3,3)
  # plt.title('LIN')
  # plt.imshow(np.absolute(linimgs[0,0]))
  # plt.colorbar()  
  # plt.show()
  # plt.close()

  # print('Sizes')
  # print('csimgs:',csimgs.shape)
  # print('orig: ',orig.shape)
  # print('linimgs: ',linimgs.shape)

  # Init vectors for MSE: one for every sample
  mse_mag_cs  = np.zeros(csimgs.shape[0])
  mse_ang_cs  = np.zeros(csimgs.shape[0])
  mse_cmx_cs  = np.zeros(csimgs.shape[0])
  perc_err_cs = np.zeros(csimgs.shape[0])

  if(addlinearrec):
    
    # Init vectors for linear MSE
    mse_mag_lin  = np.zeros(csimgs.shape[0])
    mse_ang_lin  = np.zeros(csimgs.shape[0])
    mse_cmx_lin  = np.zeros(csimgs.shape[0])
    perc_err_lin = np.zeros(csimgs.shape[0])

    # Compute image average
    avglin = linimgs.mean(axis=0,keepdims=True)
    
  for k in range(csimgs.shape[0]):

    # Magnitude
    mse_mag_cs[k]  = get_mse_mag(k,csimgs,refimg,useFluidMask,fluidMask)
    # Angle - Now that you mention it, I think we should add np.angle(orig)-np.angle(csimgs[k]) and np.abs(orig-csimgs[k]).
    mse_ang_cs[k]  = get_mse_ang(k,csimgs,refimg,useFluidMask,fluidMask)
    # Complex
    mse_cmx_cs[k]  = get_mse_cmx(k,csimgs,refimg,useFluidMask,fluidMask)
    # Percent Error
    perc_err_cs[k] = get_perc_diff(k,csimgs,refimg,useFluidMask,fluidMask)

    if(addlinearrec):
      
      # Magnitude
      mse_mag_lin[k] = get_mse_mag(k,linimgs,refimg,useFluidMask,fluidMask)
      # Angle
      mse_ang_lin[k] = get_mse_ang(k,linimgs,refimg,useFluidMask,fluidMask)
      # Complex
      mse_cmx_lin[k] = get_mse_cmx(k,linimgs,refimg,useFluidMask,fluidMask)
      # Percent Error
      perc_err_lin[k] = get_perc_diff(k,csimgs,refimg,useFluidMask,fluidMask)

  if(addlinearrec):
    return mse_mag_cs, mse_ang_cs, mse_cmx_cs, perc_err_cs, mse_mag_lin, mse_ang_lin, mse_cmx_lin, perc_err_lin
  else:
    return mse_mag_cs, mse_ang_cs, mse_cmx_cs, perc_err_cs, None, None, None, None

def plot_pdiff(args,perc_dict):

  print('Plotting MSE for various undersampling ratios...')

  # Set the baseline conditions
  # These are the first element of the lists
  bl_noise  = args.noise[0]
  bl_uval   = args.uval[0]
  bl_utype  = args.utype[0]
  bl_method = args.method[0]
  bl_wavelet = args.wavelet[0]

  if(args.singlechannel):
    numChannels = 1
  else:
    numChannels = 4

  for ch in range(numChannels):

    # CS    
    allplt_mag = [None]*len(args.uval)
    allplt_ang = [None]*len(args.uval)
    allplt_cmx = [None]*len(args.uval)
    # Linear
    allplt_mag_lin = [None]*len(args.uval)
    allplt_ang_lin = [None]*len(args.uval)
    allplt_cmx_lin = [None]*len(args.uval)

    for i,uval in enumerate(sorted(args.uval)): 
      
      mse_mag_cs, mse_ang_cs, mse_cmx_cs, perc_err_cs, mse_mag_lin, mse_ang_lin, mse_cmx_lin, perc_err_lin = get_error(ch, args.dir, args.maskdir, bl_noise, uval, bl_utype, bl_wavelet, bl_method, args.numsamples, args.usecompleximgs, args.usetrueimg, args.addlinearrec, args.usefluidmask, args.fluidmaskfile, args.usemultipatterns)
      
      allplt_mag[i] = mse_mag_cs
      allplt_ang[i] = mse_ang_cs
      allplt_cmx[i] = mse_cmx_cs

      if(args.addlinearrec):
        allplt_mag_lin[i] = mse_mag_lin
        allplt_ang_lin[i] = mse_ang_lin
        allplt_cmx_lin[i] = mse_cmx_lin

      if(args.usetrueimg):
        msg = 'tru'
        ylabel = 'MSE - w.r.t. true image'
      else:
        msg = 'avg'
        ylabel = 'MSE - w.r.t. average image'

      if(args.usecompleximgs):
        msg += '_cmx'
      else:
        msg += '_vel'

      # Store percent error
      comp = 'pdiff_k'+str(ch)+'_p'+str(uval)+'_'+msg
      perc_dict[comp+'_cs'] = perc_err_cs
      perc_dict[comp+'_lin'] = perc_err_lin

    lgd = [str(int(x*100)) + '\%' for x in sorted(args.uval)]
    fig = generateBarPlot(lgd,'Undersampling ratio',ylabel,
                          args.usecompleximgs,args.addlinearrec,
                          allplt_mag,allplt_ang,allplt_cmx,
                          allplt_mag_lin,allplt_ang_lin,allplt_cmx_lin)

    fname = args.outputdir + 'pdiff_' + msg + '_noise' + str(int(bl_noise*100)) + '_' + bl_utype + '_k' + str(ch) + '.png'
    plt.savefig(fname,dpi=200)
    print("MSE Image saved: " + fname)
    plt.close(fig)

def plot_noisediff(args,perc_dict):
  '''
  Ploting MSE for variable k-space noise
  '''

  print('Plotting MSE for various noise intensities...')

  # Set the baseline conditions
  # These are the first element of the lists
  bl_noise  = args.noise[0]
  bl_uval   = args.uval[0]
  bl_utype  = args.utype[0]
  bl_method = args.method[0]
  bl_wavelet = args.wavelet[0]

  if(args.singlechannel):
    numChannels = 1
  else:
    numChannels = 4

  for ch in range(numChannels):

    # CS    
    allplt_mag = [None]*len(args.noise)
    allplt_ang = [None]*len(args.noise)
    allplt_cmx = [None]*len(args.noise)
    # Linear
    allplt_mag_lin = [None]*len(args.noise)
    allplt_ang_lin = [None]*len(args.noise)
    allplt_cmx_lin = [None]*len(args.noise)

    for i,noise in enumerate(sorted(args.noise)):
      
      mse_mag_cs, mse_ang_cs, mse_cmx_cs, perc_err_cs, mse_mag_lin, mse_ang_lin, mse_cmx_lin, perc_err_lin = get_error(ch, args.dir, args.maskdir, noise, bl_uval, bl_utype, bl_wavelet, bl_method, args.numsamples, args.usecompleximgs, args.usetrueimg, args.addlinearrec, args.usefluidmask, args.fluidmaskfile, args.usemultipatterns)

      allplt_mag[i] = mse_mag_cs
      allplt_ang[i] = mse_ang_cs
      allplt_cmx[i] = mse_cmx_cs

      if(args.addlinearrec):
        allplt_mag_lin[i] = mse_mag_lin
        allplt_ang_lin[i] = mse_ang_lin
        allplt_cmx_lin[i] = mse_cmx_lin
        
      if(args.usetrueimg):
        msg = 'tru'
        ylabel = 'MSE - w.r.t. true image'
      else:
        msg = 'avg'
        ylabel = 'MSE - w.r.t. average image'

      if(args.usecompleximgs):
        msg += '_cmx'
      else:
        msg += '_vel'

      # Store percent error
      comp = 'noisediff_k'+str(ch)+'_noise'+str(noise)+'_'+msg
      perc_dict[comp+'_cs'] = perc_err_cs
      perc_dict[comp+'_lin'] = perc_err_lin

    # Generate Plots
    lgd = [str(int(x*100)) + '\%' for x in sorted(args.noise)]
    # fig = generateViolinPlot(lgd,'Noise intensity',allplt,allplt_lin)
    fig = generateBarPlot(lgd,'Noise intensity',ylabel,
                          args.usecompleximgs,args.addlinearrec,
                          allplt_mag,allplt_ang,allplt_cmx,
                          allplt_mag_lin,allplt_ang_lin,allplt_cmx_lin)

    # Save Img  
    fname = args.outputdir + 'noisediff_' + msg + '_p' + str(int(bl_uval*100)) + '_' + bl_utype + '_k' + str(ch) + '.png'
    plt.savefig(fname,dpi=200)
    print("MSE Image saved: " + fname)
    plt.close(fig)

def plot_maskdiff(args,perc_dict):
  '''
  Ploting MSE for variable undesampling masks
  '''
  print('Plotting MSE for various undersampling masks...')

  # Set the baseline conditions
  # These are the first element of the lists
  bl_noise  = args.noise[0]
  bl_uval   = args.uval[0]
  bl_utype  = args.utype[0]
  bl_method = args.method[0]
  bl_wavelet = args.wavelet[0]

  if(args.singlechannel):
    numChannels = 1
  else:
    numChannels = 4

  for ch in range(numChannels):

    # CS    
    allplt_mag = [None]*len(args.utype)
    allplt_ang = [None]*len(args.utype)
    allplt_cmx = [None]*len(args.utype)
    # Linear
    allplt_mag_lin = [None]*len(args.utype)
    allplt_ang_lin = [None]*len(args.utype)
    allplt_cmx_lin = [None]*len(args.utype)

    for i,utype in enumerate(args.utype):
      
      mse_mag_cs, mse_ang_cs, mse_cmx_cs, perc_err_cs, mse_mag_lin, mse_ang_lin, mse_cmx_lin, perc_err_lin = get_error(ch,args.dir, args.maskdir, bl_noise, bl_uval, utype, bl_wavelet, bl_method, args.numsamples, args.usecompleximgs, args.usetrueimg, args.addlinearrec, args.usefluidmask, args.fluidmaskfile, args.usemultipatterns)
      
      allplt_mag[i] = mse_mag_cs
      allplt_ang[i] = mse_ang_cs
      allplt_cmx[i] = mse_cmx_cs

      if(args.addlinearrec):
        allplt_mag_lin[i] = mse_mag_lin
        allplt_ang_lin[i] = mse_ang_lin
        allplt_cmx_lin[i] = mse_cmx_lin

      if(args.usetrueimg):
        msg = 'tru'
        ylabel = 'MSE - w.r.t. true image'
      else:
        msg = 'avg'
        ylabel = 'MSE - w.r.t. average image'

      if(args.usecompleximgs):
        msg += '_cmx'
      else:
        msg += '_vel'

      # Store percent error
      comp = 'maskdiff_k'+str(ch)+'_'+str(utype)+'_'+msg
      perc_dict[comp+'_cs'] = perc_err_cs
      perc_dict[comp+'_lin'] = perc_err_lin

    lgd = []
    for loopA in range(len(args.utype)):
      lgd.append(get_umask_string(args.utype[loopA]))

    # fig = generateViolinPlot(lgd,'Undersampling mask',allplt,allplt_lin)    
    fig = generateBarPlot(lgd,'Undersampling mask',ylabel,
                          args.usecompleximgs,args.addlinearrec,
                          allplt_mag,allplt_ang,allplt_cmx,
                          allplt_mag_lin,allplt_ang_lin,allplt_cmx_lin,rotext=True)

    # Save Img  
    fname = args.outputdir + 'maskdiff_' + msg + '_noise' + str(int(bl_noise*100)) + '_p' + str(int(bl_uval*100)) + '_' + bl_utype + '_k' + str(ch) + '.png'
    plt.savefig(fname,dpi=200)
    print("MSE Image saved: " + fname)
    plt.close(fig)  

def plot_methoddiff(args,perc_dict):

  print('Plotting MSE for various reconstruction methods...')
    
  # Set the baseline conditions
  # These are the first element of the lists
  bl_noise  = args.noise[0]
  bl_uval   = args.uval[0]
  bl_utype  = args.utype[0]
  bl_method = args.method[0]
  bl_wavelet = args.wavelet[0]

  if(args.singlechannel):
    numChannels = 1
  else:
    numChannels = 4

  for ch in range(numChannels):

    # CS    
    allplt_mag = [None]*len(args.method)
    allplt_ang = [None]*len(args.method)
    allplt_cmx = [None]*len(args.method)
    # Linear
    allplt_mag_lin = [None]*len(args.method)
    allplt_ang_lin = [None]*len(args.method)
    allplt_cmx_lin = [None]*len(args.method)

    for i,method in enumerate(args.method):

      mse_mag_cs, mse_ang_cs, mse_cmx_cs, perc_err_cs, mse_mag_lin, mse_ang_lin, mse_cmx_lin, perc_err_lin = get_error(ch,args.dir, args.maskdir, bl_noise, bl_uval, bl_utype, bl_wavelet, method, args.numsamples, args.usecompleximgs, args.usetrueimg, args.addlinearrec, args.usefluidmask, args.fluidmaskfile, args.usemultipatterns)
        
      allplt_mag[i] = mse_mag_cs
      allplt_ang[i] = mse_ang_cs
      allplt_cmx[i] = mse_cmx_cs

      if(args.addlinearrec):
        allplt_mag_lin[i] = mse_mag_lin
        allplt_ang_lin[i] = mse_ang_lin
        allplt_cmx_lin[i] = mse_cmx_lin

      if(args.usetrueimg):
        msg = 'tru'
        ylabel = 'MSE - w.r.t. true image'
      else:
        msg = 'avg'
        ylabel = 'MSE - w.r.t. average image'

      if(args.usecompleximgs):
        msg += '_cmx'
      else:
        msg += '_vel'

      # Store percent error
      comp = 'methoddiff_k'+str(ch)+'_'+str(method)+'_'+msg
      perc_dict[comp+'_cs'] = perc_err_cs
      perc_dict[comp+'_lin'] = perc_err_lin        

    # Generate Plots
    lgd = []
    for loopA in range(len(args.method)):
      lgd.append(get_method_string(args.method[loopA]))

    # fig = generateViolinPlot(lgd,'Reconstruction method',allplt,allplt_lin)    
    fig = generateBarPlot(lgd,'Reconstruction method',ylabel,
                          args.usecompleximgs,args.addlinearrec,
                          allplt_mag,allplt_ang,allplt_cmx,
                          allplt_mag_lin,allplt_ang_lin,allplt_cmx_lin,rotext=True)

    fname = args.outputdir + 'methoddiff_' + msg + '_noise' + str(int(bl_noise*100)) + '_p'+str(int(bl_uval*100)) + '_' + bl_utype + '_k' + str(ch) + '.png'
    plt.savefig(fname,dpi=200)
    print("MSE Image saved: " + fname)
    plt.close(fig)

# Difference in the wavelet frame
def plot_waveletdiff(args,perc_dict):

  print('Plotting MSE for various wavelet frames...')
    
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

  for ch in range(numChannels):

    # CS    
    allplt_mag = [None]*len(args.wavelet)
    allplt_ang = [None]*len(args.wavelet)
    allplt_cmx = [None]*len(args.wavelet)
    # Linear
    allplt_mag_lin = [None]*len(args.wavelet)
    allplt_ang_lin = [None]*len(args.wavelet)
    allplt_cmx_lin = [None]*len(args.wavelet)

    for i,wavelet in enumerate(args.wavelet):

      mse_mag_cs, mse_ang_cs, mse_cmx_cs, perc_err_cs, mse_mag_lin, mse_ang_lin, mse_cmx_lin, perc_err_lin = get_error(ch,args.dir, args.maskdir, bl_noise, bl_uval, bl_utype, wavelet, bl_method, args.numsamples, args.usecompleximgs, args.usetrueimg, args.addlinearrec, args.usefluidmask, args.fluidmaskfile, args.usemultipatterns)
        
      allplt_mag[i] = mse_mag_cs
      allplt_ang[i] = mse_ang_cs
      allplt_cmx[i] = mse_cmx_cs

      if(args.addlinearrec):
        allplt_mag_lin[i] = mse_mag_lin
        allplt_ang_lin[i] = mse_ang_lin
        allplt_cmx_lin[i] = mse_cmx_lin

      if(args.usetrueimg):
        msg = 'tru'
        ylabel = 'MSE - w.r.t. true image'
      else:
        msg = 'avg'
        ylabel = 'MSE - w.r.t. average image'

      if(args.usecompleximgs):
        msg += '_cmx'
      else:
        msg += '_vel'

      # Store percent error
      comp = 'wavediff_k'+str(ch)+'_'+str(wavelet)+'_'+msg
      perc_dict[comp+'_cs'] = perc_err_cs
      perc_dict[comp+'_lin'] = perc_err_lin                

    # Generate Plots
    lgd = []
    for loopA in range(len(args.wavelet)):
      lgd.append(get_wavelet_string(args.wavelet[loopA]))

    # fig = generateViolinPlot(lgd,'Reconstruction method',allplt,allplt_lin)    
    fig = generateBarPlot(lgd,'Wavelet Frame',ylabel,
                          args.usecompleximgs,args.addlinearrec,
                          allplt_mag,allplt_ang,allplt_cmx,
                          allplt_mag_lin,allplt_ang_lin,allplt_cmx_lin,rotext=True)

    fname = args.outputdir + 'wavediff_' + msg + '_noise' + str(int(bl_noise*100)) + '_p'+str(int(bl_uval*100)) + '_' + bl_utype + '_k' + str(ch) + '.png'
    plt.savefig(fname,dpi=200)
    print("MSE Image saved: " + fname)
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

  # wavelet
  parser.add_argument('-w', '--wavelet',
                      action=None,
                      nargs='*',
                      const=None,
                      default=['haar'],
                      type=str,
                      choices=['haar','db8'],
                      required=False,
                      help='list of wavelet frames',
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
  # maskdir
  parser.add_argument('-k', '--maskdir',
                      action=None,
                      # nargs='+',
                      const=None,
                      default='./',
                      type=str,
                      choices=None,
                      required=False,
                      help='folder containing the undersampling masks',
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
  # usecompleximgs
  parser.add_argument('--usecompleximgs',
                      action='store_true',
                      default=False,
                      required=False,
                      help='evaluate the MSE based on the reconstructed complex image. Default: use velocity components.',
                      dest='usecompleximgs')    

  # usefluidmask
  parser.add_argument('--usefluidmask',
                      action='store_true',
                      default=False,
                      required=False,
                      help='evaluate the MSE only within the fluid region. Default: use full image',
                      dest='usefluidmask')    

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

  # addlinear reconstructions
  parser.add_argument('--addlinearrec',
                      action='store_true',
                      default=False,
                      required=False,
                      help='Plot linear reconstruction MSE.',
                      dest='addlinearrec')      
  # use true image
  parser.add_argument('--usetrueimg',
                      action='store_true',
                      default=False,
                      required=False,
                      help='use the true image to evaluate the MSE',
                      dest='usetrueimg')    

  # use unique undesampling pattern for each reconstruction (false) or change every time
  parser.add_argument('-um', '--usemultipatterns',
                      action='store_true',
                      default=False,
                      required=False,
                      help='generate a unique undersampling pattern for each noise realization',
                      dest='usemultipatterns')

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

  # String for perc_err files
  parser.add_argument('-ps', '--percstring',
                      action=None,
                      # nargs='*',
                      const=None,
                      default='',
                      type=str,
                      choices=None,
                      required=False,
                      help='string for percent error file',
                      metavar='',
                      dest='percstring')
  

  # Parse Commandline Arguments
  args = parser.parse_args()

  # Plot the parameter perturbations
  perc_dict = {}
  if(len(args.noise) > 1):
    plot_noisediff(args,perc_dict)
  if(len(args.uval) > 1):
    plot_pdiff(args,perc_dict)
  if(len(args.utype) > 1):
    plot_maskdiff(args,perc_dict)
  if(len(args.method) > 1):
    plot_methoddiff(args,perc_dict)
  if(len(args.method) > 1):
    plot_waveletdiff(args,perc_dict)

  # Save Dictionary to File
  perc_file = 'perc_dict_' + args.percstring + '.npy'
  print('Saving percent error dictionary to file: ',perc_file)
  np.save(perc_file, perc_dict)

  # Completed!
  if(args.printlevel > 0):
    print('Completed!!!')  
