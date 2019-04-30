#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File I/O demo for radial multicoil brain data
March 2019
By: By: Martin Uecker (martin.uecker@med.uni-goettingen.de) and Florian Knoll (florian.knoll@nyumc.org)

Demo data for reproducible research study group initiative to reproduce [1]

[1] Pruessmann, K. P.; Weiger, M.; Boernert, P. and Boesiger, P.
Advances in sensitivity encoding with arbitrary k-space trajectories.
Magn Reson Med 46: 638-651 (2001)
"""
#%reset

import h5py
import numpy as np
import matplotlib.pyplot as plt

from bart import bart

plt.close("all")

#%% Load data
h5_dataset = h5py.File('rawdata_brain_radial_96proj_12ch.h5', 'r')
#h5_dataset = h5py.File('rawdata_heart_radial_55proj_34ch.h5', 'r')
print("Keys: %s" % h5_dataset.keys())
h5_dataset_rawdata_name = list(h5_dataset.keys())[0]
h5_dataset_trajectory_name = list(h5_dataset.keys())[1]

trajectory = h5_dataset.get(h5_dataset_trajectory_name).value
rawdata = h5_dataset.get(h5_dataset_rawdata_name).value

[dummy,nFE,nSpokes,nCh] = rawdata.shape

#%% Display rawdata and trajectory
plt.figure(1)
plt.imshow(np.log(1+np.abs(rawdata[0,:,:,0])),cmap="gray")
plt.axis('off')
plt.title('rawdata coil 1')

#%% Subsample
#R = 2
#trajectory = trajectory[:,:,1::R]
#rawdata = rawdata[:,:,1::R,:]
#[dummy,nFE,nSpokes,nCh] = rawdata.shape

#%%  Demo: NUFFT reconstruction with BART
# inverse gridding
img_igrid = bart(1, 'nufft -i -t', trajectory, rawdata)

# channel combination
img_igrid_sos = bart(1, 'rss 8', img_igrid)
img_igrid_sos = np.abs(img_igrid_sos)

#%% Display results
plt.figure(2)
plt.imshow(np.fliplr(np.flipud(img_igrid_sos)),cmap="gray")
plt.axis('off')
plt.title('Regridding SOS reconstruction')