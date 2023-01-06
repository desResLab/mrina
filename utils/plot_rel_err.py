import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

fs = 14
plt.rc('font',  family='serif')
plt.rc('xtick', labelsize='x-small')
plt.rc('ytick', labelsize='x-small')
plt.rc('text',  usetex=True)

x_labels = ['noise = 1\%',
            'noise = 5\%',
            'noise = 10\%',
            'noise = 30\%',
            '$m_{k}/m$ = 25\%',
            '$m_{k}/m$ = 50\%',
            '$m_{k}/m$ = 75\%',
            #'Gauss',
            #'Bernoulli',
            'CS',
            'CS Debias',
            'OMP',
            'Haar',
            'DB8']

# Velocity w.r.t. the true velocity
dict_tru = np.load('perc_dict_3.npy',allow_pickle=True)
# Velocity w.r.t. the average velocity
dict_avg = np.load('perc_dict_4.npy',allow_pickle=True)

# Collect the values to be plotted
res = np.zeros((12,4,2))
for dir_id in range(4):
  for diff in ['tru','avg']:
    if(diff == 'tru'):
      diff_id = 0
      curr_dict = dict_tru.item()
    else:
      diff_id = 1
      curr_dict = dict_avg.item()

    # Get arrays 
    res[0,dir_id,diff_id] = np.mean(curr_dict['noisediff_k'+str(dir_id)+'_noise0.01_'+diff+'_vel_cs'])
    res[1,dir_id,diff_id] = np.mean(curr_dict['noisediff_k'+str(dir_id)+'_noise0.05_'+diff+'_vel_cs'])
    res[2,dir_id,diff_id] = np.mean(curr_dict['noisediff_k'+str(dir_id)+'_noise0.1_'+diff+'_vel_cs'])
    res[3,dir_id,diff_id] = np.mean(curr_dict['noisediff_k'+str(dir_id)+'_noise0.3_'+diff+'_vel_cs'])

    res[4,dir_id,diff_id] = np.mean(curr_dict['pdiff_k'+str(dir_id)+'_p0.25_'+diff+'_vel_cs'])
    res[5,dir_id,diff_id] = np.mean(curr_dict['pdiff_k'+str(dir_id)+'_p0.5_'+diff+'_vel_cs'])
    res[6,dir_id,diff_id] = np.mean(curr_dict['pdiff_k'+str(dir_id)+'_p0.75_'+diff+'_vel_cs'])

    # res[7,dir_id,diff_id] = np.mean(curr_dict['maskdiff_k'+str(dir_id)+'_vardengauss_'+diff+'_vel_cs'])
    # res[8,dir_id,diff_id] = np.mean(curr_dict['maskdiff_k'+str(dir_id)+'_bernoulli_'+diff+'_vel_cs'])

    res[7,dir_id,diff_id] = np.mean(curr_dict['methoddiff_k'+str(dir_id)+'_cs_'+diff+'_vel_cs'])
    res[8,dir_id,diff_id] = np.mean(curr_dict['methoddiff_k'+str(dir_id)+'_csdebias_'+diff+'_vel_cs'])
    res[9,dir_id,diff_id] = np.mean(curr_dict['methoddiff_k'+str(dir_id)+'_omp_'+diff+'_vel_cs'])

    res[10,dir_id,diff_id] = np.mean(curr_dict['wavediff_k'+str(dir_id)+'_haar_'+diff+'_vel_cs'])
    res[11,dir_id,diff_id] = np.mean(curr_dict['wavediff_k'+str(dir_id)+'_db8_'+diff+'_vel_cs'])

fig = plt.figure(figsize=(4,4))
# Wrt true image
plt.semilogy(np.arange(12),res[:,0,0]*100,'o-',ls='-.',c='r',mew=1.5,mfc='none',label='k = 0 - true')
plt.semilogy(np.arange(12),res[:,1,0]*100,'o-',ls='-.',c='b',mew=1.5,mfc='none',label='k = 1 - true')
plt.semilogy(np.arange(12),res[:,2,0]*100,'o-',ls='-.',c='m',mew=1.5,mfc='none',label='k = 2 - true')
plt.semilogy(np.arange(12),res[:,3,0]*100,'o-',ls='-.',c='c',mew=1.5,mfc='none',label='k = 3 - true')
# Wrt avg image
plt.semilogy(np.arange(12),res[:,0,1]*100,'^-',ls=':',c='r',mew=1.5,mfc='none',label='k = 0 - avg')
plt.semilogy(np.arange(12),res[:,1,1]*100,'^-',ls=':',c='b',mew=1.5,mfc='none',label='k = 1 - avg')
plt.semilogy(np.arange(12),res[:,2,1]*100,'^-',ls=':',c='m',mew=1.5,mfc='none',label='k = 2 - avg')
plt.semilogy(np.arange(12),res[:,3,1]*100,'^-',ls=':',c='c',mew=1.5,mfc='none',label='k = 3 - avg')

if(False):
  plt.legend(ncol=1, loc='top left', labelspacing=1.39, bbox_to_anchor=(1, 1.03))
plt.xticks(np.arange(12), x_labels, rotation=70, fontsize=fs, ha='right')
plt.yticks(fontsize=fs)
plt.title('Slice 1 Test Case',fontsize=fs)
plt.grid(visible=True, which='both', axis='both', ls=':', c='gray',alpha=0.4)
plt.ylabel('Error [\%]',fontsize=fs)
rect = plt.Rectangle((-0.5,0.00001), 4, 100000, fc='gray',alpha=0.4)
fig.gca().add_patch(rect)
rect = plt.Rectangle((6.5,0.00001), 3, 100000, fc='gray',alpha=0.4)
fig.gca().add_patch(rect)
fig.gca().yaxis.set_major_formatter(FormatStrFormatter('%.f'))
plt.ylim([0,1000])
plt.xlim([-0.5,11.5])
plt.tight_layout()
# plt.show()
plt.savefig('sl_perc_err.pdf')