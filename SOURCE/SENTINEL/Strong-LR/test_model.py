#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 14:44:41 2019

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
test_data = np.load(os.path.join(config.NUMPY_DIR, "sentinel_test_data.npy"))
num_features = 12
# %%
learning_rate = 0.001
n_epochs = 10000

tf.reset_default_graph()
with tf.name_scope('data'):
    X = tf.placeholder(tf.float32, [None, num_features], name="inputs")
    Y = tf.placeholder(tf.float32, [None, 1], name="labels")

with tf.variable_scope("Variables", reuse=tf.AUTO_REUSE):
    W = tf.get_variable("Weights", [num_features, 1], initializer=tf.contrib.layers.xavier_initializer())
    b = tf.get_variable("Biases", [1], initializer=tf.zeros_initializer())

Z = tf.matmul(X, W, name="multiply_weights")
Z = tf.add(Z, b, name="add_bias")
Z = tf.nn.sigmoid(Z)
#%%
print("TEST MODEL")
saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, os.path.join(config.MODEL_DIR, "SENTINEL", "Strong-LR", "model.ckpt"))
    data = test_data[:,:num_features]
    feed_dict = {X: data}
    preds = sess.run(Z, feed_dict=feed_dict)

pred_labels = np.zeros(preds.shape)
pred_labels[preds > 0.5] = 1
pred_labels[preds < 0.5] = 0
labels = np.reshape(test_data[:, -1], [-1, 1])
print("Accuracy:", len(np.where([pred_labels == labels])[1])/int(len(labels)))
print("f1_score:", f1_score(labels, pred_labels))
plt.hist(preds[labels==0], color="red")
plt.hist(preds[labels==1], color="green")
plt.show()
