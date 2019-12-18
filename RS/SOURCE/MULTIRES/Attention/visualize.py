#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 10:42:06 2019

@author: ghosh128
"""

import sys
sys.path.append("../../")
import os
import config
import numpy as np
import itertools
from PIL import Image
import tensorflow as tf
from sklearn.metrics import f1_score
from osgeo import gdal, gdalconst, osr
tf.set_random_seed(1)
# %%
# print("LOAD DATA")
# im_size = (3660,3660)
# landsat_data = np.load(os.path.join(config.NUMPY_DIR, "LANDSAT_data.npy"))
# landsat_data = landsat_data[:int(im_size[0]/2), int(im_size[1]/2):, :]
# im_size = (10980,10980)
# sentinel_data = np.load(os.path.join(config.NUMPY_DIR, "SENTINEL_data.npy"))
# sentinel_data = sentinel_data[:int(im_size[0]/2), int(im_size[1]/2):, :]
# num_features_fine = 12
# num_features_coarse = 11
# # %%
# print("BUILD MODEL")
# tf.reset_default_graph()
# with tf.name_scope('data'):
#     X_fine_consistency= tf.placeholder(tf.float32, [9*1830, num_features_fine], name="fine_res_consistency_inputs")
#     X_coarse_consistency = tf.placeholder(tf.float32, [1830, num_features_coarse], name="coarse_res_consistency_inputs")
#
# with tf.variable_scope("Variables", reuse=tf.AUTO_REUSE):
#     W_1_fine = tf.get_variable("Weights_layer_1_fine", [num_features_fine, 6], initializer=tf.contrib.layers.xavier_initializer())
#     b_1_fine = tf.get_variable("Biases_layer_1_fine", [6], initializer=tf.zeros_initializer())
#     W_2_fine = tf.get_variable("Weights_layer_2_fine", [6, 1], initializer=tf.contrib.layers.xavier_initializer())
#     b_2_fine = tf.get_variable("Biases_layer_2_fine", [1], initializer=tf.zeros_initializer())
#
#     W_1_coarse = tf.get_variable("Weights_layer_1_coarse", [num_features_coarse, 6], initializer=tf.contrib.layers.xavier_initializer())
#     b_1_coarse = tf.get_variable("Biases_layer_1_coarse", [6], initializer=tf.zeros_initializer())
#     W_2_coarse = tf.get_variable("Weights_layer_2_coarse", [6, 1], initializer=tf.contrib.layers.xavier_initializer())
#     b_2_coarse = tf.get_variable("Biases_layer_2_coarse", [1], initializer=tf.zeros_initializer())
#
#     W_attention = tf.get_variable("Weights_attention", [12, 1], initializer=tf.contrib.layers.xavier_initializer())
#
# visualize_att_weights = []
#
# Z_fine_consistency = tf.nn.sigmoid(tf.add(tf.matmul(X_fine_consistency, W_1_fine, name="multiply_weights"), b_1_fine, name="add_bias"))
# Z_coarse_consistency = tf.nn.sigmoid(tf.add(tf.matmul(X_coarse_consistency, W_1_coarse, name="multiply_weights"), b_1_coarse, name="add_bias"))
# ind = list(itertools.chain.from_iterable(itertools.repeat(x, 9) for x in range(1830)))
# Z_concat_consistency = tf.concat([tf.gather(Z_coarse_consistency, ind), Z_fine_consistency], axis=-1)
# Z_attention_consistency = []
# for i in range(1830):
#     score = tf.matmul(tf.nn.tanh(Z_concat_consistency[i*9:(i+1)*9]), W_attention)
#     attention_weights = tf.nn.softmax(score, axis=0)
#     visualize_att_weights.append(attention_weights)
#     context = tf.divide(tf.matmul(tf.transpose(attention_weights), Z_fine_consistency[i*9:(i+1)*9]), tf.reduce_sum(attention_weights))
#     Z_attention_consistency.append(context)
# Z_attention_consistency = tf.reshape(tf.convert_to_tensor(Z_attention_consistency), (1830, Z_fine_consistency.shape[1]))
# Z_coarse_consistency = tf.nn.sigmoid(tf.add(tf.matmul(Z_coarse_consistency, W_2_coarse, name="multiply_weights"), b_2_coarse, name="add_bias"))
# Z_fine_consistency = tf.nn.sigmoid(tf.add(tf.matmul(Z_attention_consistency, W_2_fine, name="multiply_weights"), b_2_fine, name="add_bias"))
#
# visualize_att_weights = tf.reshape(tf.convert_to_tensor(visualize_att_weights), (1830, -1))
# print(visualize_att_weights.shape)
# # %%
# arr_att_weights = np.zeros(sentinel_data.shape[:-1])
# print("VISUALIZE MODEL")
# saver = tf.train.Saver()
# with tf.Session() as sess:
#     saver.restore(sess, os.path.join(config.MODEL_DIR, "MULTI_RES", "Attention", "model.ckpt"))
#     for i in range(landsat_data.shape[0]):
#         if not i%10:
#             print(i)
#         visualize_data_coarse = np.zeros((1830, 11))
#         visualize_data_fine = np.zeros((1830*9, 12))
#         for j in range(landsat_data.shape[1]):
#             visualize_data_coarse[j,:] = landsat_data[i, j, :]
#             visualize_data_fine[j*9:(j+1)*9, :] = np.reshape(sentinel_data[i*3:(i+1)*3, j*3:(j+1)*3, :], (-1,12))
#         feed_dict = {X_fine_consistency: visualize_data_fine, X_coarse_consistency: visualize_data_coarse}
#         attention_weights_arr = sess.run(visualize_att_weights, feed_dict=feed_dict)
#         attention_weights_arr = np.reshape(attention_weights_arr, (1830,3,3))
#         arr_att_weights[i*3:(i+1)*3,:] = np.transpose(np.reshape(attention_weights_arr,(-1,3)))
#
# np.save("Attention_array_visualize_1", arr_att_weights)
#
# print(np.min(arr_att_weights))
# print(np.max(arr_att_weights))
# arr = np.zeros(im_size)
# arr[:int(im_size[0]/2), int(im_size[1]/2):] = arr_att_weights
# arr = arr.astype(np.float32)
#
# print(np.min(arr))
# print(np.max(arr))
#
# filename = os.path.join("Attention_array_visualize_1.tif")
# tif_with_meta = gdal.Open(os.path.join(config.OSM_DIR, 'SENTINEL.tif'), gdalconst.GA_ReadOnly)
# gt = tif_with_meta.GetGeoTransform()
# driver = gdal.GetDriverByName("GTiff")
# dest = driver.Create(filename, 10980, 10980, 1, gdal.GDT_Float64)
# dest.GetRasterBand(1).WriteArray(arr)
# dest.SetGeoTransform(gt)
# wkt = tif_with_meta.GetProjection()
# srs = osr.SpatialReference()
# srs.ImportFromWkt(wkt)
# dest.SetProjection(srs.ExportToWkt())
# dest = None
# #%%
im_size = (10980,10980)
arr_att_weights = np.load("Attention_array_visualize_1.npy")
print(np.min(arr_att_weights))
print(np.max(arr_att_weights))
ds = gdal.Open("Attention_array_visualize_1.tif")
rb = ds.GetRasterBand(1)
arr_att_weights = rb.ReadAsArray()
print(np.min(arr_att_weights))
print(np.max(arr_att_weights))
#%%
# arr = np.zeros(im_size)
# arr[:int(im_size[0]/2), int(im_size[1]/2):] = arr_att_weights
# arr = arr.astype(np.float32)
#
# filename = os.path.join("Attention_array_visualize.tif")
# tif_with_meta = gdal.Open(os.path.join(config.OSM_DIR, 'SENTINEL.tif'), gdalconst.GA_ReadOnly)
# gt = tif_with_meta.GetGeoTransform()
# driver = gdal.GetDriverByName("GTiff")
# dest = driver.Create(filename, 10980, 10980, 1, gdal.GDT_Float64)
# dest.GetRasterBand(1).WriteArray(arr)
# dest.SetGeoTransform(gt)
# wkt = tif_with_meta.GetProjection()
# srs = osr.SpatialReference()
# srs.ImportFromWkt(wkt)
# dest.SetProjection(srs.ExportToWkt())
# dest = None
