import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# ====
# MAIN
# ====
if __name__ == "__main__":

  fs=12
  plt.rc('font', family='serif')
  plt.rc('xtick', labelsize='x-small')
  plt.rc('ytick', labelsize='x-small')
  plt.rc('text', usetex=True)

  num = '03'

  fig = plt.figure(figsize=(10,3))
  # for num in range(1,4):
  if(True):
    sl = np.load('imgs_n1_'+str(num).zfill(2)+'.npy')
    mask = np.load('slice'+str(num).zfill(2)+'_mask.npy')
    x,y = np.meshgrid(np.arange(sl.shape[3]),np.arange(sl.shape[4]))
    u,v,w  = sl[0,1,0],sl[0,2,0],sl[0,3,0]
    # plt.subplot(1,3,num)
    plt.imshow(sl[0,0,0],alpha=0.6)
    plt.imshow(mask,alpha=0.6)
    qv = plt.quiver(x, y, u, v, w, scale=3000)
    cax = fig.gca().inset_axes([1.04, 0.0, 0.05, 1.0])
    cb = plt.colorbar(qv,cax=cax)
    for t in cb.ax.get_yticklabels():
      t.set_fontsize(10)
    cb.set_label('Out of plane velocity [cm/s]',fontsize=fs)
    plt.axis(False)
    plt.title('Slice '+str(num),fontsize=fs)
    plt.tight_layout()
    # plt.show()

  plt.savefig('slice_'+str(num)+'.pdf',bbox_inches='tight', pad_inches=0)
  plt.close()



