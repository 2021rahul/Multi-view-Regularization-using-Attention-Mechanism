#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 14:27:29 2019

@author: ghosh128
"""

import sys
sys.path.append("../../")
import os
import config
import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
tf.set_random_seed(1)
# %%
print("LOAD DATA")
validate_data = np.load(os.path.join(config.NUMPY_DIR, "landsat_validate_data.npy"))
num_features = 11
# %%
print("BUILD MODEL")
tf.reset_default_graph()
with tf.name_scope('data'):
    X = tf.placeholder(tf.float32, [None, num_features], name="inputs")
    Y = tf.placeholder(tf.float32, [None, 1], name="labels")

with tf.variable_scope("Variables", reuse=tf.AUTO_REUSE):
    W_1 = tf.get_variable("Weights_layer_1", [num_features, 6], initializer=tf.contrib.layers.xavier_initializer())
    b_1 = tf.get_variable("Biases_layer_1", [6], initializer=tf.zeros_initializer())
    W_2 = tf.get_variable("Weights_layer_2", [6, 1], initializer=tf.contrib.layers.xavier_initializer())
    b_2 = tf.get_variable("Biases_layer_2", [1], initializer=tf.zeros_initializer())

Z = tf.nn.sigmoid(tf.add(tf.matmul(X, W_1, name="multiply_weights"), b_1, name="add_bias"))
Z = tf.nn.sigmoid(tf.add(tf.matmul(Z, W_2, name="multiply_weights"), b_2, name="add_bias"))
#%%
print("VALIDATE MODEL")
saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, os.path.join(config.MODEL_DIR, "LANDSAT", "OnlyStrong", "model.ckpt"))
    data = validate_data[:,:num_features]
    feed_dict = {X: data}
    preds = sess.run(Z, feed_dict=feed_dict)

pred_labels = np.zeros(preds.shape)
pred_labels[preds > 0.5] = 1
pred_labels[preds < 0.5] = 0
labels = np.reshape(validate_data[:, -1], [-1, 1])
print("Accuracy:", len(np.where([pred_labels == labels])[1])/int(len(labels)))
print("f1_score:", f1_score(labels, pred_labels))
plt.hist(preds[labels==0], color="red")
plt.hist(preds[labels==1], color="green")
plt.show()
