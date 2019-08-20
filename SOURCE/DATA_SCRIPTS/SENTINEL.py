#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 07:44:14 2019

@author: rahul2021
"""

import os
from PIL import Image
import numpy as np
import random
#%%
DATA_DIR = "../../DATA/ROME"
SENTINEL_DIR = os.path.join(DATA_DIR, "SENTINEL")
OSM_DIR = os.path.join(DATA_DIR, "OSM")
OUT_DIR = os.path.join(DATA_DIR, "NUMPY")
if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)
im_size = (10980,10980)
#%%
osm_im = Image.open(os.path.join(OSM_DIR, "SENTINEL.tif"))
osm_im_array = np.array(osm_im)
#%%
im = Image.open(os.path.join(SENTINEL_DIR, "T32TQM_20190322T100031_B01_60m.jp2"))
B1 = np.array(im.resize(im_size))

im = Image.open(os.path.join(SENTINEL_DIR, "T32TQM_20190322T100031_B02_10m.jp2"))
B2 = np.array(im.resize(im_size))

im = Image.open(os.path.join(SENTINEL_DIR, "T32TQM_20190322T100031_B03_10m.jp2"))
B3 = np.array(im.resize(im_size))

im = Image.open(os.path.join(SENTINEL_DIR, "T32TQM_20190322T100031_B04_10m.jp2"))
B4 = np.array(im.resize(im_size))

im = Image.open(os.path.join(SENTINEL_DIR, "T32TQM_20190322T100031_B05_20m.jp2"))
B5 = np.array(im.resize(im_size))

im = Image.open(os.path.join(SENTINEL_DIR, "T32TQM_20190322T100031_B06_20m.jp2"))
B6 = np.array(im.resize(im_size))

im = Image.open(os.path.join(SENTINEL_DIR, "T32TQM_20190322T100031_B07_20m.jp2"))
B7 = np.array(im.resize(im_size))

im = Image.open(os.path.join(SENTINEL_DIR, "T32TQM_20190322T100031_B08_10m.jp2"))
B8 = np.array(im.resize(im_size))

im = Image.open(os.path.join(SENTINEL_DIR, "T32TQM_20190322T100031_B8A_20m.jp2"))
B8A = np.array(im.resize(im_size))

im = Image.open(os.path.join(SENTINEL_DIR, "T32TQM_20190322T100031_B09_60m.jp2"))
B9 = np.array(im.resize(im_size))

im = Image.open(os.path.join(SENTINEL_DIR, "T32TQM_20190322T100031_B11_20m.jp2"))
B11 = np.array(im.resize(im_size))

im = Image.open(os.path.join(SENTINEL_DIR, "T32TQM_20190322T100031_B12_20m.jp2"))
B12 = np.array(im.resize(im_size))

data = np.stack((B1,B2,B3,B4,B5,B6,B7,B8,B8A,B9,B11,B12), axis=-1)
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
np.save(os.path.join(OUT_DIR, "sentinel_train_data"), train_data)
np.save(os.path.join(OUT_DIR, "sentinel_test_data"), test_data)






