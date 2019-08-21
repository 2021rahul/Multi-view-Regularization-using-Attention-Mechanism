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
import tensorflow as tf
tf.set_random_seed(1)
# %%
print("LOAD DATA")
consistency_data_fine = np.load(os.path.join(config.NUMPY_DIR, "sentinel_consistency_data.npy"))
consistency_data_coarse = np.load(os.path.join(config.NUMPY_DIR, "landsat_consistency_data.npy"))
num_features_fine = 12
num_features_coarse = 11
# %%
print("BUILD MODEL")
tf.reset_default_graph()
with tf.name_scope('data'):
    X_fine_consistency= tf.placeholder(tf.float32, [9, 12], name="fine_res_consistency_inputs")
    X_coarse_consistency = tf.placeholder(tf.float32, [1, 11], name="coarse_res_consistency_inputs")

with tf.variable_scope("Variables", reuse=tf.AUTO_REUSE):
    W_attention = tf.get_variable("Weights_attention", [num_features_coarse+num_features_fine, 1], initializer=tf.contrib.layers.xavier_initializer())

ind = list(itertools.chain.from_iterable(itertools.repeat(x, 9) for x in range(1)))
X_concat_consistency = tf.concat([tf.gather(X_coarse_consistency, ind), X_fine_consistency], axis=-1)
score = tf.nn.tanh(tf.matmul(X_concat_consistency, W_attention))
#attention_weights = tf.nn.softmax(score, axis=0)
attention_weights = tf.pow(config.x, score)/tf.reduce_sum(tf.pow(config.x, score))
#%%
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    consistency_coarse = np.reshape(consistency_data_coarse[0, :], (1,-1))
    consistency_fine = np.reshape(consistency_data_fine[:9, :], (9,-1))
    feed_dict = {X_fine_consistency: consistency_fine, X_coarse_consistency: consistency_coarse}
    att_weights, score_arr = sess.run([attention_weights, score], feed_dict=feed_dict)
    print(np.sum(att_weights))