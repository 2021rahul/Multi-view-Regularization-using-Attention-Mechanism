#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 07:53:43 2019

@author: rahul2021
"""

import sys
sys.path.append("../../")
import os
import config
import numpy as np
import random

if not os.path.exists(config.NUMPY_DIR):
    os.makedirs(config.NUMPY_DIR)
if not os.path.exists(config.RESULT_DIR):
    os.makedirs(config.RESULT_DIR)
if not os.path.exists(config.MODEL_DIR):
    os.makedirs(config.MODEL_DIR)
# %%
im_size = (3660,3660)
landsat_data = np.load(os.path.join(config.NUMPY_DIR, "LANDSAT_data.npy"))
landsat_data = landsat_data[int(im_size[0]/2):, int(im_size[1]/2):, :]
# %%
im_size = (10980,10980)
sentinel_data = np.load(os.path.join(config.NUMPY_DIR, "SENTINEL_data.npy"))
sentinel_data = sentinel_data[int(im_size[0]/2):, int(im_size[1]/2):, :]
#%%
N = config.num_consistency
index_i = random.sample(range(landsat_data.shape[0]), N)
index_j = random.sample(range(landsat_data.shape[1]), N)

landsat = np.zeros((N*N, 11))
sentinel = np.zeros((N*N*9, 12))
for i,landsat_i in enumerate(index_i):
    for j,landsat_j in enumerate(index_j):
        landsat[i*N+j,:] = landsat_data[landsat_i, landsat_j, :]
        sentinel[(i*N+j)*9:(i*N+j+1)*9, :] = np.reshape(sentinel_data[landsat_i*3:(landsat_i+1)*3, landsat_j*3:(landsat_j+1)*3, :], (-1,12))

print(landsat.shape)
print(sentinel.shape)
np.save(os.path.join(config.NUMPY_DIR, "landsat_consistency_data"), landsat)
np.save(os.path.join(config.NUMPY_DIR, "sentinel_consistency_data"), sentinel)
