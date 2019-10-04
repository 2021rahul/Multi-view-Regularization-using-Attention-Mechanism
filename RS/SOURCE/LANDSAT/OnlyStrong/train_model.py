#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 14:35:17 2019

@author: ghosh128
"""

import sys
sys.path.append("../../")
import os
import config
import numpy as np
import tensorflow as tf
tf.set_random_seed(1)
# %%
print("LOAD DATA")
train_data = np.load(os.path.join(config.NUMPY_DIR, "landsat_train_data.npy"))
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

with tf.name_scope("loss_function"):
    loss = tf.reduce_mean(tf.losses.mean_squared_error(predictions=Z, labels=Y))
tf.summary.scalar('loss', loss)

global_step = tf.Variable(0, name='global_step', trainable=False)
with tf.variable_scope("optimizer", reuse=tf.AUTO_REUSE):
    optimizer = tf.train.AdamOptimizer(config.SENTINEL_OnlyStrong_learning_rate).minimize(loss, global_step)
# %%
print("TRAIN MODEL")
saver = tf.train.Saver()
merged_summary_op = tf.summary.merge_all()
with tf.Session() as sess:
    summary_writer = tf.summary.FileWriter(os.path.join(config.MODEL_DIR, "LANDSAT", "OnlyStrong"), sess.graph)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    for i in range(config.SENTINEL_OnlyStrong_n_epochs):
        data = train_data[:, :num_features]
        label = np.reshape(train_data[:, -1], [-1,1])
        feed_dict = {X: data, Y: label}
        summary_str, _, loss_epoch = sess.run([merged_summary_op, optimizer, loss], feed_dict=feed_dict)
        summary_writer.add_summary(summary_str, global_step=global_step.eval())
        if not i%100:
            print('Epoch: {0} Loss: {1}'.format(i, loss_epoch))
    summary_writer.close()
    save_path = saver.save(sess, os.path.join(config.MODEL_DIR, "LANDSAT", "OnlyStrong", "model.ckpt"))