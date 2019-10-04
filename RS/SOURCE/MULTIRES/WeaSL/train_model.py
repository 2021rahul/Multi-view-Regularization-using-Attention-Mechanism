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
import itertools
import tensorflow as tf
tf.set_random_seed(1)
# %%
print("LOAD DATA")
train_data_fine = np.load(os.path.join(config.NUMPY_DIR, "sentinel_train_data.npy"))
train_data_coarse = np.load(os.path.join(config.NUMPY_DIR, "landsat_train_data.npy"))
consistency_data_fine = np.load(os.path.join(config.NUMPY_DIR, "sentinel_consistency_data.npy"))
consistency_data_coarse = np.load(os.path.join(config.NUMPY_DIR, "landsat_consistency_data.npy"))
num_features_fine = 12
num_features_coarse = 11
# %%
print("BUILD COARSE MODEL")
tf.reset_default_graph()
with tf.name_scope('data'):
    X_coarse = tf.placeholder(tf.float32, [None, num_features_coarse], name="corase_res_inputs")
    Y_coarse = tf.placeholder(tf.float32, [None, 1], name="coarse_res_labels")

with tf.variable_scope("Variables", reuse=tf.AUTO_REUSE):
    W_1_coarse = tf.get_variable("Weights_layer_1_coarse", [num_features_coarse, 6], initializer=tf.contrib.layers.xavier_initializer())
    b_1_coarse = tf.get_variable("Biases_layer_1_coarse", [6], initializer=tf.zeros_initializer())
    W_2_coarse = tf.get_variable("Weights_layer_2_coarse", [6, 1], initializer=tf.contrib.layers.xavier_initializer())
    b_2_coarse = tf.get_variable("Biases_layer_2_coarse", [1], initializer=tf.zeros_initializer())

Z_coarse = tf.nn.sigmoid(tf.add(tf.matmul(X_coarse, W_1_coarse, name="multiply_weights"), b_1_coarse, name="add_bias"))
Z_coarse = tf.nn.sigmoid(tf.add(tf.matmul(Z_coarse, W_2_coarse, name="multiply_weights"), b_2_coarse, name="add_bias"))

with tf.name_scope("loss_function"):
    loss = tf.reduce_mean(tf.losses.mean_squared_error(predictions=Z_coarse, labels=Y_coarse))
tf.summary.scalar('loss', loss)

global_step = tf.Variable(0, name='global_step', trainable=False)
with tf.variable_scope("optimizer", reuse=tf.AUTO_REUSE):
    optimizer = tf.train.AdamOptimizer(config.MULTIRES_WeaSL_learning_rate).minimize(loss, global_step)
# %%
print("TRAIN COARSE MODEL")
saver = tf.train.Saver()
merged_summary_op = tf.summary.merge_all()
with tf.Session() as sess:
    summary_writer = tf.summary.FileWriter(os.path.join(config.MODEL_DIR, "MULTI_RES", "WeaSL"), sess.graph)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    for i in range(config.MULTIRES_WeaSL_n_epochs):
        data = train_data_coarse[:, :num_features_coarse]
        label = np.reshape(train_data_coarse[:, -1], [-1,1])
        feed_dict = {X_coarse: data, Y_coarse: label}
        summary_str, _, loss_epoch = sess.run([merged_summary_op, optimizer, loss], feed_dict=feed_dict)
        summary_writer.add_summary(summary_str, global_step=global_step.eval())
        if not i%100:
            print('Epoch: {0} Loss: {1}'.format(i, loss_epoch))
    summary_writer.close()
    save_path = saver.save(sess, os.path.join(config.MODEL_DIR, "MULTI_RES", "WeaSL", "model_coarse.ckpt"))
# %%
print("GET COARSE PREDICTIONS")
saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, os.path.join(config.MODEL_DIR, "MULTI_RES", "WeaSL", "model_coarse.ckpt"))
    feed_dict = {X_coarse: consistency_data_coarse}
    preds_coarse = sess.run(Z_coarse, feed_dict=feed_dict)

pred_labels_coarse = np.zeros(preds_coarse.shape)
pred_labels_coarse[preds_coarse > 0.5] = 1
pred_labels_coarse[preds_coarse < 0.5] = 0
# %%
print("PREPARE DATA")
consistency_data = np.hstack((consistency_data_fine, np.reshape(np.repeat(pred_labels_coarse, 9), (-1,1))))
data = np.hstack((train_data_fine[:,:num_features_fine], np.reshape(train_data_fine[:, -1], [-1,1])))
train_data = np.vstack((consistency_data, data))
# %%
print("BUILD FINE MODEL")
tf.reset_default_graph()
with tf.name_scope('data'):
    X_fine = tf.placeholder(tf.float32, [None, num_features_fine], name="fine_res_inputs")
    Y_fine = tf.placeholder(tf.float32, [None, 1], name="fine_res_labels")

with tf.variable_scope("Variables", reuse=tf.AUTO_REUSE):
    W_1_fine = tf.get_variable("Weights_layer_1_fine", [num_features_fine, 6], initializer=tf.contrib.layers.xavier_initializer())
    b_1_fine = tf.get_variable("Biases_layer_1_fine", [6], initializer=tf.zeros_initializer())
    W_2_fine = tf.get_variable("Weights_layer_2_fine", [6, 1], initializer=tf.contrib.layers.xavier_initializer())
    b_2_fine = tf.get_variable("Biases_layer_2_fine", [1], initializer=tf.zeros_initializer())

Z_fine = tf.nn.sigmoid(tf.add(tf.matmul(X_fine, W_1_fine, name="multiply_weights"), b_1_fine, name="add_bias"))
Z_fine = tf.nn.sigmoid(tf.add(tf.matmul(Z_fine, W_2_fine, name="multiply_weights"), b_2_fine, name="add_bias"))

with tf.name_scope("loss_function"):
    loss = tf.reduce_mean(tf.losses.mean_squared_error(predictions=Z_fine, labels=Y_fine))
tf.summary.scalar('loss', loss)

global_step = tf.Variable(0, name='global_step', trainable=False)
with tf.variable_scope("optimizer", reuse=tf.AUTO_REUSE):
    optimizer = tf.train.AdamOptimizer(config.MULTIRES_WeaSL_learning_rate).minimize(loss, global_step)
# %%
print("TRAIN MODEL")
saver = tf.train.Saver()
merged_summary_op = tf.summary.merge_all()
with tf.Session() as sess:
    summary_writer = tf.summary.FileWriter(os.path.join(config.MODEL_DIR, "MULTI_RES", "WeaSL"), sess.graph)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    k=0
    for i in range(config.MULTIRES_WeaSL_n_epochs):
        if (k+1)*config.MULTIRES_WeaSL_batch_consistency>len(train_data):
            k = 0
        data = train_data[(k*9)*config.MULTIRES_WeaSL_batch_consistency:((k+1)*9)*config.MULTIRES_WeaSL_batch_consistency, :]

        features = data[:, :-1]
        label = np.reshape(data[:, -1], [-1,1])

        feed_dict = {X_fine: features, Y_fine: label}
        summary_str, _, loss_epoch = sess.run([merged_summary_op, optimizer, loss], feed_dict=feed_dict)
        summary_writer.add_summary(summary_str, global_step=global_step.eval())
        if not i%100:
            print('Epoch: {0} Loss: {1}'.format(i, loss_epoch))
    summary_writer.close()
    save_path = saver.save(sess, os.path.join(config.MODEL_DIR, "MULTI_RES", "WeaSL", "model_fine.ckpt"))
