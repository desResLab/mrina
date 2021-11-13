import numpy as np

p1_noise01 = np.load('01_poiseuilleaxis1/snr_noise1_n100.npy')
p1_noise05 = np.load('01_poiseuilleaxis1/snr_noise5_n100.npy')
p1_noise10 = np.load('01_poiseuilleaxis1/snr_noise10_n100.npy')
p1_noise30 = np.load('01_poiseuilleaxis1/snr_noise30_n100.npy')

p2_noise01 = np.load('02_poiseuilleaxis2/snr_noise1_n100.npy')
p2_noise05 = np.load('02_poiseuilleaxis2/snr_noise5_n100.npy')
p2_noise10 = np.load('02_poiseuilleaxis2/snr_noise10_n100.npy')
p2_noise30 = np.load('02_poiseuilleaxis2/snr_noise30_n100.npy')

ia_noise01 = np.load('03_idealaorta/snr_noise1_n100.npy')
ia_noise05 = np.load('03_idealaorta/snr_noise5_n100.npy')
ia_noise10 = np.load('03_idealaorta/snr_noise10_n100.npy')
ia_noise30 = np.load('03_idealaorta/snr_noise30_n100.npy')

ma_noise00 = np.load('04_aortamri/snr_noise0_n100.npy')
ma_noise10 = np.load('05_aortamri_n10/snr_noise10_n100.npy')

print(p1_noise01.max(),p1_noise05.max(),p1_noise10.max(),p1_noise30.max())

print(p2_noise01.max(),p2_noise05.max(),p2_noise10.max(),p2_noise30.max())

print(ia_noise01.max(),ia_noise05.max(),ia_noise10.max(),ia_noise30.max())

print(ma_noise00.max(),ma_noise10.max())


