#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 11:06:48 2019

@author: ghosh128
"""

import sys
sys.path.append("../../")
import os
import config
from PIL import Image
import numpy as np
import random

if not os.path.exists(config.NUMPY_DIR):
    os.makedirs(config.NUMPY_DIR)
if not os.path.exists(config.RESULT_DIR):
    os.makedirs(config.RESULT_DIR)
if not os.path.exists(config.MODEL_DIR):
    os.makedirs(config.MODEL_DIR)
## Region1 Region2
## Region3 Region4
landsat_channel = 11
sentinel_channel = 12
#%%
sentinel_im_size = (10980,10980)
sentinel_data = np.load(os.path.join(config.NUMPY_DIR, "SENTINEL_data.npy"))
osm_im = Image.open(os.path.join(config.OSM_DIR, "SENTINEL_label.tif"))
osm_im_array = np.array(osm_im)

landsat_im_size = (3660,3660)
landsat_data = np.load(os.path.join(config.NUMPY_DIR, "LANDSAT_data.npy"))
# %%
# Region 4 : int(im_size[0]/2):, int(im_size[1]/2):
landsat_data_region = landsat_data[int(landsat_im_size[0]/2):, int(landsat_im_size[1]/2):, :]
label_region = osm_im_array[int(sentinel_im_size[0]/2):, int(sentinel_im_size[1]/2):]
sentinel_data_region = sentinel_data[int(sentinel_im_size[0]/2):, int(sentinel_im_size[1]/2):, :]

index_0_i, index_0_j = np.where(label_region==0)
index = random.sample(range(len(index_0_i)), len(index_0_i))
index_0_i = index_0_i[index]
index_0_j = index_0_j[index]
index_1_i, index_1_j = np.where(label_region==1)
index = random.sample(range(len(index_1_i)), len(index_1_i))
index_1_i = index_1_i[index]
index_1_j = index_1_j[index]

train_index_0_i = index_0_i[:config.sentinel_num_train]
train_index_0_j = index_0_j[:config.sentinel_num_train]
train_index_1_i = index_1_i[:config.sentinel_num_train]
train_index_1_j = index_1_j[:config.sentinel_num_train]
landsat = np.zeros((config.sentinel_num_train, 11))
sentinel = np.zeros((config.sentinel_num_train, 12))
for index in range(config.sentinel_num_train):
    sentinel_i, sentinel_j = train_index_0_i[index], train_index_0_j[index]
    sentinel[index, :] = sentinel_data_region[sentinel_i, sentinel_j, :]
    landsat[index,:] = landsat_data_region[int(sentinel_i/3), int(sentinel_j/3), :]
data_0 = np.concatenate((sentinel, landsat, np.zeros((config.sentinel_num_train, 1))), axis = 1)
landsat = np.zeros((config.sentinel_num_train, 11))
sentinel = np.zeros((config.sentinel_num_train, 12))
for index in range(config.sentinel_num_train):
    sentinel_i, sentinel_j = train_index_1_i[index], train_index_1_j[index]
    sentinel[index, :] = sentinel_data_region[sentinel_i, sentinel_j, :]
    landsat[index,:] = landsat_data_region[int(sentinel_i/3), int(sentinel_j/3), :]
data_1 = np.concatenate((landsat, sentinel, np.ones((config.sentinel_num_train, 1))), axis = 1)
train_data = np.concatenate((data_0, data_1))
index = random.sample(range(2*config.sentinel_num_train), 2*config.sentinel_num_train)
train_data = train_data[index]

validate_index_0_i = index_0_i[config.sentinel_num_train:config.sentinel_num_train+config.sentinel_num_validate]
validate_index_0_j = index_0_j[config.sentinel_num_train:config.sentinel_num_train+config.sentinel_num_validate]
validate_index_1_i = index_1_i[config.sentinel_num_train:config.sentinel_num_train+config.sentinel_num_validate]
validate_index_1_j = index_1_j[config.sentinel_num_train:config.sentinel_num_train+config.sentinel_num_validate]
landsat = np.zeros((config.sentinel_num_validate, 11))
sentinel = np.zeros((config.sentinel_num_validate, 12))
for index in range(config.sentinel_num_validate):
    sentinel_i, sentinel_j = validate_index_0_i[index], validate_index_0_j[index]
    sentinel[index, :] = sentinel_data_region[sentinel_i, sentinel_j, :]
    landsat[index,:] = landsat_data_region[int(sentinel_i/3), int(sentinel_j/3), :]
data_0 = np.concatenate((sentinel, landsat, np.zeros((config.sentinel_num_validate, 1))), axis = 1)
landsat = np.zeros((config.sentinel_num_validate, 11))
sentinel = np.zeros((config.sentinel_num_validate, 12))
for index in range(config.sentinel_num_validate):
    sentinel_i, sentinel_j = validate_index_1_i[index], validate_index_1_j[index]
    sentinel[index, :] = sentinel_data_region[sentinel_i, sentinel_j, :]
    landsat[index,:] = landsat_data_region[int(sentinel_i/3), int(sentinel_j/3), :]
data_1 = np.concatenate((landsat, sentinel, np.ones((config.sentinel_num_validate, 1))), axis = 1)
validate_data = np.concatenate((data_0, data_1))
index = random.sample(range(2*config.sentinel_num_validate), 2*config.sentinel_num_validate)
validate_data = validate_data[index]

print(train_data.shape)
print(validate_data.shape)
np.save(os.path.join(config.NUMPY_DIR, "sentinel_train_data"), train_data)
np.save(os.path.join(config.NUMPY_DIR, "sentinel_validate_data"), validate_data)
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
