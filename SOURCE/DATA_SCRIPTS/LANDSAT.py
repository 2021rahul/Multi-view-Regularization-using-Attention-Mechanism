#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 20:57:40 2019

@author: rahul2021
"""

import os
from PIL import Image
import numpy as np
import random
#%%
DATA_DIR = "../../DATA/ROME"
LANDSAT_DIR = os.path.join(DATA_DIR, "LANDSAT")
OSM_DIR = os.path.join(DATA_DIR, "OSM")
OUT_DIR = os.path.join(DATA_DIR, "NUMPY")
if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)
im_size = (3660,3660)
#%%
osm_im = Image.open(os.path.join(OSM_DIR, "LANDSAT.tif"))
osm_im_array = np.array(osm_im)
#%%
im = Image.open(os.path.join(LANDSAT_DIR, "B1.tif"))
B1 = np.array(im.resize(im_size))

im = Image.open(os.path.join(LANDSAT_DIR, "B2.tif"))
B2 = np.array(im.resize(im_size))

im = Image.open(os.path.join(LANDSAT_DIR, "B3.tif"))
B3 = np.array(im.resize(im_size))

im = Image.open(os.path.join(LANDSAT_DIR, "B4.tif"))
B4 = np.array(im.resize(im_size))

im = Image.open(os.path.join(LANDSAT_DIR, "B5.tif"))
B5 = np.array(im.resize(im_size))

im = Image.open(os.path.join(LANDSAT_DIR, "B6.tif"))
B6 = np.array(im.resize(im_size))

im = Image.open(os.path.join(LANDSAT_DIR, "B7.tif"))
B7 = np.array(im.resize(im_size))

im = Image.open(os.path.join(LANDSAT_DIR, "B8.tif"))
B8 = np.array(im.resize(im_size))

im = Image.open(os.path.join(LANDSAT_DIR, "B9.tif"))
B9 = np.array(im.resize(im_size))

im = Image.open(os.path.join(LANDSAT_DIR, "B10.tif"))
B10 = np.array(im.resize(im_size))

im = Image.open(os.path.join(LANDSAT_DIR, "B11.tif"))
B11 = np.array(im.resize(im_size))

data = np.stack((B1,B2,B3,B4,B5,B6,B7,B8,B9,B10,B11), axis=-1)
data = data.astype("float")
#%%
for channel in range(11):
    data[:,:,channel] = (data[:,:,channel]-np.mean(data[:,:,channel]))/np.std(data[:,:,channel])
data = np.reshape(data, (-1,11))
label = np.reshape(osm_im_array, (-1,1))
data = np.hstack((data, label))
#%%
index_0 = np.where(data[:, -1]==0)[0]
index = random.sample(range(len(index_0)), len(index_0))
index_0 = index_0[index]
index_1 = np.where(data[:, -1]==1)[0]
index = random.sample(range(len(index_1)), len(index_1))
index_1 = index_1[index]
print("Number of 0 data: ", len(index_0))
print("Number of 1 data: ", len(index_1))
#%%
train_split = int(0.5*min(len(index_0), len(index_1)))

train_index_0 = index_0[:train_split]
test_index_0 = index_0[train_split:min(len(index_0), len(index_1))]

train_index_1 = index_1[:train_split]
test_index_1 = index_1[train_split:min(len(index_0), len(index_1))]

train_index = np.concatenate((train_index_0, train_index_1))
index = random.sample(range(len(train_index)), len(train_index))
train_index = train_index[index]

test_index = np.concatenate((test_index_0, test_index_1))
index = random.sample(range(len(test_index)), len(test_index))
test_index = test_index[index]
#%%
train_data = data[train_index]
test_data = data[test_index]
np.save(os.path.join(OUT_DIR, "landsat_train_data"), train_data)
np.save(os.path.join(OUT_DIR, "landsat_test_data"), test_data)






