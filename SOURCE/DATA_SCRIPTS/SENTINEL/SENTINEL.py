#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 11:06:48 2019

@author: ghosh128
"""

import os
from PIL import Image
import numpy as np
import random

REGION = "MINNEAPOLIS"
DATA_DIR = "../../..//DATA/"+REGION
LANDSAT_DIR = os.path.join(DATA_DIR, "LANDSAT")
OSM_DIR = os.path.join(DATA_DIR, "OSM")
OUT_DIR = os.path.join(DATA_DIR, "NUMPY")
if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)
## Region1 Region2
## Region3 Region4
landsat_channel = 11
sentinel_channel = 12
num_train_samples = 100
num_test_samples = 10000
#%%
sentinel_im_size = (10980,10980)
sentinel_data = np.load(os.path.join(OUT_DIR, "SENTINEL_data.npy"))
osm_im = Image.open(os.path.join(OSM_DIR, "SENTINEL_label.tif"))
osm_im_array = np.array(osm_im)

landsat_im_size = (3660,3660)
landsat_data = np.load(os.path.join(OUT_DIR, "LANDSAT_data.npy"))
# %%
# Region 4 : int(im_size[0]/2):, int(im_size[1]/2):
landsat_data_region = landsat_data[int(landsat_im_size[0]/2):, int(landsat_im_size[1]/2):, :]
label_region = osm_im_array[int(sentinel_im_size[0]/2):, int(sentinel_im_size[1]/2):]
sentinel_data_region = sentinel_data[int(sentinel_im_size[0]/2):, int(sentinel_im_size[1]/2):, :]

index_0_i, index_0_j = np.where(label_region==0)
index = random.sample(range(len(index_0_i)), len(index_0_i))
index_0_i = index_0_i[index[:num_train_samples]]
index_0_j = index_0_j[index[:num_train_samples]]

landsat = np.zeros((num_train_samples, 11))
sentinel = np.zeros((num_train_samples, 12))
for index in range(num_train_samples):
    sentinel_i, sentinel_j = index_0_i[index], index_0_j[index]
    sentinel[index, :] = sentinel_data_region[sentinel_i, sentinel_j, :]
    landsat[index,:] = landsat_data[int(sentinel_i/3), int(sentinel_j/3), :]
data_0 = np.concatenate((sentinel, landsat, np.zeros((num_train_samples, 1))), axis = 1)

index_1_i, index_1_j = np.where(label_region==1)
index = random.sample(range(len(index_1_i)), len(index_1_i))
index_1_i = index_1_i[index[:num_train_samples]]
index_1_j = index_1_j[index[:num_train_samples]]

landsat = np.zeros((num_train_samples, 11))
sentinel = np.zeros((num_train_samples, 12))
for index in range(num_train_samples):
    sentinel_i, sentinel_j = index_1_i[index], index_1_j[index]
    sentinel[index, :] = sentinel_data_region[sentinel_i, sentinel_j, :]
    landsat[index,:] = landsat_data[int(sentinel_i/3), int(sentinel_j/3), :]
data_1 = np.concatenate((sentinel, landsat, np.ones((num_train_samples, 1))), axis = 1)

train_data = np.concatenate((data_0, data_1))
index = random.sample(range(2*num_train_samples), 2*num_train_samples)
train_data = train_data[index]
np.save(os.path.join(OUT_DIR, "sentinel_train_data_"+str(2*num_train_samples)), train_data)
##%%
## Region 1 : :int(im_size[0]/2), :int(im_size[1]/2)
#landsat_data_region = landsat_data[:int(landsat_im_size[0]/2), :int(landsat_im_size[1]/2), :]
#label_region = osm_im_array[:int(sentinel_im_size[0]/2), :int(sentinel_im_size[1]/2)]
#sentinel_data_region = sentinel_data[:int(sentinel_im_size[0]/2), :int(sentinel_im_size[1]/2), :]
#
#index_0_i, index_0_j = np.where(label_region==0)
#index = random.sample(range(len(index_0_i)), len(index_0_i))
#index_0_i = index_0_i[index[:num_test_samples]]
#index_0_j = index_0_j[index[:num_test_samples]]
#
#landsat = np.zeros((num_test_samples, 11))
#sentinel = np.zeros((num_test_samples, 12))
#for index in range(num_test_samples):
#    sentinel_i, sentinel_j = index_0_i[index], index_0_j[index]
#    sentinel[index, :] = sentinel_data_region[sentinel_i, sentinel_j, :]
#    landsat[index,:] = landsat_data[int(sentinel_i/3), int(sentinel_j/3), :]
#data_0 = np.concatenate((sentinel, landsat, np.zeros((num_test_samples, 1))), axis = 1)
#
#index_1_i, index_1_j = np.where(label_region==1)
#index = random.sample(range(len(index_1_i)), len(index_1_i))
#index_1_i = index_1_i[index[:num_test_samples]]
#index_1_j = index_1_j[index[:num_test_samples]]
#
#landsat = np.zeros((num_test_samples, 11))
#sentinel = np.zeros((num_test_samples, 12))
#for index in range(num_test_samples):
#    sentinel_i, sentinel_j = index_1_i[index], index_1_j[index]
#    sentinel[index, :] = sentinel_data_region[sentinel_i, sentinel_j, :]
#    landsat[index,:] = landsat_data[int(sentinel_i/3), int(sentinel_j/3), :]
#data_1 = np.concatenate((sentinel, landsat, np.ones((num_test_samples, 1))), axis = 1)
#
#test_data_1 = np.concatenate((data_0, data_1))
#index = random.sample(range(2*num_test_samples), 2*num_test_samples)
#test_data_1 = test_data_1[index]
#
## Region 2 : :int(im_size[0]/2), int(im_size[1]/2):
#landsat_data_region = landsat_data[:int(landsat_im_size[0]/2), int(landsat_im_size[1]/2):, :]
#label_region = osm_im_array[:int(sentinel_im_size[0]/2), int(sentinel_im_size[1]/2):]
#sentinel_data_region = sentinel_data[:int(sentinel_im_size[0]/2), int(sentinel_im_size[1]/2):, :]
#
#index_0_i, index_0_j = np.where(label_region==0)
#index = random.sample(range(len(index_0_i)), len(index_0_i))
#index_0_i = index_0_i[index[:num_test_samples]]
#index_0_j = index_0_j[index[:num_test_samples]]
#
#landsat = np.zeros((num_test_samples, 11))
#sentinel = np.zeros((num_test_samples, 12))
#for index in range(num_test_samples):
#    sentinel_i, sentinel_j = index_0_i[index], index_0_j[index]
#    sentinel[index, :] = sentinel_data_region[sentinel_i, sentinel_j, :]
#    landsat[index,:] = landsat_data[int(sentinel_i/3), int(sentinel_j/3), :]
#data_0 = np.concatenate((sentinel, landsat, np.zeros((num_test_samples, 1))), axis = 1)
#
#index_1_i, index_1_j = np.where(label_region==1)
#index = random.sample(range(len(index_1_i)), len(index_1_i))
#index_1_i = index_1_i[index[:num_test_samples]]
#index_1_j = index_1_j[index[:num_test_samples]]
#
#landsat = np.zeros((num_test_samples, 11))
#sentinel = np.zeros((num_test_samples, 12))
#for index in range(num_test_samples):
#    sentinel_i, sentinel_j = index_1_i[index], index_1_j[index]
#    sentinel[index, :] = sentinel_data_region[sentinel_i, sentinel_j, :]
#    landsat[index,:] = landsat_data[int(sentinel_i/3), int(sentinel_j/3), :]
#data_1 = np.concatenate((sentinel, landsat, np.ones((num_test_samples, 1))), axis = 1)
#
#test_data_2 = np.concatenate((data_0, data_1))
#index = random.sample(range(2*num_test_samples), 2*num_test_samples)
#test_data_2 = test_data_2[index]
#
## Region 3 : int(im_size[0]/2):, :int(im_size[1]/2)
#if REGION != "ROME":
#    landsat_data_region = landsat_data[int(landsat_im_size[0]/2):, :int(landsat_im_size[1]/2), :]
#    label_region = osm_im_array[int(sentinel_im_size[0]/2):, :int(sentinel_im_size[1]/2)]
#    sentinel_data_region = sentinel_data[int(sentinel_im_size[0]/2):, :int(sentinel_im_size[1]/2), :]
#
#    index_0_i, index_0_j = np.where(label_region==0)
#    index = random.sample(range(len(index_0_i)), len(index_0_i))
#    index_0_i = index_0_i[index[:num_test_samples]]
#    index_0_j = index_0_j[index[:num_test_samples]]
#
#    landsat = np.zeros((num_test_samples, 11))
#    sentinel = np.zeros((num_test_samples, 12))
#    for index in range(num_test_samples):
#        sentinel_i, sentinel_j = index_0_i[index], index_0_j[index]
#        sentinel[index, :] = sentinel_data_region[sentinel_i, sentinel_j, :]
#        landsat[index,:] = landsat_data[int(sentinel_i/3), int(sentinel_j/3), :]
#    data_0 = np.concatenate((sentinel, landsat, np.zeros((num_test_samples, 1))), axis = 1)
#
#    index_1_i, index_1_j = np.where(label_region==1)
#    index = random.sample(range(len(index_1_i)), len(index_1_i))
#    index_1_i = index_1_i[index[:num_test_samples]]
#    index_1_j = index_1_j[index[:num_test_samples]]
#    
#    landsat = np.zeros((num_test_samples, 11))
#    sentinel = np.zeros((num_test_samples, 12))
#    for index in range(num_test_samples):
#        sentinel_i, sentinel_j = index_1_i[index], index_1_j[index]
#        sentinel[index, :] = sentinel_data_region[sentinel_i, sentinel_j, :]
#        landsat[index,:] = landsat_data[int(sentinel_i/3), int(sentinel_j/3), :]
#    data_1 = np.concatenate((sentinel, landsat, np.ones((num_test_samples, 1))), axis = 1)
#
#    test_data_3 = np.concatenate((data_0, data_1))
#    index = random.sample(range(2*num_test_samples), 2*num_test_samples)
#    test_data_3 = test_data_3[index]
#
#if REGION == "ROME":
#    test_data = np.concatenate((test_data_1, test_data_2))
#else:
#    test_data = np.concatenate((test_data_1, test_data_2, test_data_3))
#np.save(os.path.join(OUT_DIR, "sentinel_test_data"), test_data)
