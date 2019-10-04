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
train_data_fine = np.load(os.path.join(config.NUMPY_DIR, "sentinel_train_data.npy"))
train_data_coarse = np.load(os.path.join(config.NUMPY_DIR, "landsat_train_data.npy"))
consistency_data_fine = np.load(os.path.join(config.NUMPY_DIR, "sentinel_consistency_data.npy"))
consistency_data_coarse = np.load(os.path.join(config.NUMPY_DIR, "landsat_consistency_data.npy"))
num_features_fine = 12
num_features_coarse = 11
# %%
print("BUILD MODEL")
tf.reset_default_graph()
with tf.name_scope('data'):
    X_fine = tf.placeholder(tf.float32, [None, num_features_fine], name="fine_res_inputs")
    Y_fine = tf.placeholder(tf.float32, [None, 1], name="fine_res_labels")
    X_coarse = tf.placeholder(tf.float32, [None, num_features_coarse], name="corase_res_inputs")
    Y_coarse = tf.placeholder(tf.float32, [None, 1], name="coarse_res_labels")

with tf.variable_scope("Variables", reuse=tf.AUTO_REUSE):
    W_1_fine = tf.get_variable("Weights_layer_1_fine", [num_features_fine, 6], initializer=tf.contrib.layers.xavier_initializer())
    b_1_fine = tf.get_variable("Biases_layer_1_fine", [6], initializer=tf.zeros_initializer())
    W_2_fine = tf.get_variable("Weights_layer_2_fine", [6, 1], initializer=tf.contrib.layers.xavier_initializer())
    b_2_fine = tf.get_variable("Biases_layer_2_fine", [1], initializer=tf.zeros_initializer())

    W_1_coarse = tf.get_variable("Weights_layer_1_coarse", [num_features_coarse, 6], initializer=tf.contrib.layers.xavier_initializer())
    b_1_coarse = tf.get_variable("Biases_layer_1_coarse", [6], initializer=tf.zeros_initializer())
    W_2_coarse = tf.get_variable("Weights_layer_2_coarse", [6, 1], initializer=tf.contrib.layers.xavier_initializer())
    b_2_coarse = tf.get_variable("Biases_layer_2_coarse", [1], initializer=tf.zeros_initializer())

Z_fine = tf.nn.sigmoid(tf.add(tf.matmul(X_fine, W_1_fine, name="multiply_weights"), b_1_fine, name="add_bias"))
Z_fine = tf.nn.sigmoid(tf.add(tf.matmul(Z_fine, W_2_fine, name="multiply_weights"), b_2_fine, name="add_bias"))

Z_coarse = tf.nn.sigmoid(tf.add(tf.matmul(X_coarse, W_1_coarse, name="multiply_weights"), b_1_coarse, name="add_bias"))
Z_coarse = tf.nn.sigmoid(tf.add(tf.matmul(Z_coarse, W_2_coarse, name="multiply_weights"), b_2_coarse, name="add_bias"))

with tf.name_scope("loss_function"):
    loss_fine = tf.reduce_mean(tf.losses.mean_squared_error(predictions=Z_fine, labels=Y_fine))
    loss_coarse = tf.reduce_mean(tf.losses.mean_squared_error(predictions=Z_coarse, labels=Y_coarse))
tf.summary.scalar('loss_fine', loss_fine)
tf.summary.scalar('loss_coarse', loss_coarse)

global_step = tf.Variable(0, name='global_step', trainable=False)
with tf.variable_scope("optimizer", reuse=tf.AUTO_REUSE):
    optimizer_coarse = tf.train.AdamOptimizer(config.MULTIRES_Augment_learning_rate).minimize(loss_coarse, global_step)
    optimizer_fine = tf.train.AdamOptimizer(config.MULTIRES_Augment_learning_rate).minimize(loss_fine, global_step)
# %%
print("TRAIN MODEL")
saver = tf.train.Saver()
merged_summary_op = tf.summary.merge_all()
with tf.Session() as sess:
    summary_writer = tf.summary.FileWriter(os.path.join(config.MODEL_DIR, "MULTI_RES", "Augment"), sess.graph)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    k=0
    for i in range(config.MULTIRES_Augment_n_epochs):
        data_fine = train_data_fine[:, :num_features_fine]
        label_fine = np.reshape(train_data_fine[:, -1], [-1,1])
        data_coarse = train_data_coarse[:, :num_features_coarse]
        label_coarse = np.reshape(train_data_coarse[:, -1], [-1,1])
        feed_dict = {X_fine:data_fine, Y_fine:label_fine, X_coarse:data_coarse, Y_coarse:label_coarse}
        summary_str, _, _, loss_fine_epoch, loss_coarse_epoch = sess.run([merged_summary_op, optimizer_coarse, optimizer_fine, loss_fine, loss_coarse], feed_dict=feed_dict)
        summary_writer.add_summary(summary_str, global_step=global_step.eval())
        if not i%100:
            print('Epoch:{0} Loss_fine:{1:.4f} Loss_coarse:{2:.4f}'.format(i, loss_fine_epoch, loss_coarse_epoch))
        k = k+1
    summary_writer.close()
    save_path = saver.save(sess, os.path.join(config.MODEL_DIR, "MULTI_RES", "Augment", "model.ckpt"))
# %%
print("PREDICT UNLABELLED REGION")
saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, os.path.join(config.MODEL_DIR, "MULTI_RES", "Augment", "model.ckpt"))
    data_fine = consistency_data_fine
    data_coarse = consistency_data_coarse
    feed_dict = {X_fine: data_fine, X_coarse: data_coarse}
    preds_fine, preds_coarse = sess.run([Z_fine, Z_coarse], feed_dict=feed_dict)

pred_labels_fine = np.zeros(preds_fine.shape)
pred_labels_fine[preds_fine > 0.5] = 1
pred_labels_fine[preds_fine < 0.5] = 0

pred_labels_coarse = np.zeros(preds_coarse.shape)
pred_labels_coarse[preds_coarse > 0.5] = 1
pred_labels_coarse[preds_coarse < 0.5] = 0
# %%
print("AUGMENT DATA")
match=np.maximum.reduceat(pred_labels_fine, np.unique(np.repeat(range(len(pred_labels_coarse)), 9), return_index=True)[1])
coarse_valid_inds = np.where(pred_labels_coarse==match)[0]
fine_valid_inds = [list(range(x*9,(x+1)*9)) for x in coarse_valid_inds]
fine_valid_inds = [item for sublist in fine_valid_inds for item in sublist]
print(len(coarse_valid_inds), len(fine_valid_inds))

augment_data_coarse = np.hstack((consistency_data_coarse[coarse_valid_inds,:], pred_labels_coarse[coarse_valid_inds,:]))
data = np.hstack((train_data_coarse[:,:num_features_coarse], np.reshape(train_data_coarse[:, -1], [-1,1])))
train_data_coarse = np.vstack((augment_data_coarse, data))

augment_data_fine = np.hstack((consistency_data_fine[fine_valid_inds,:], pred_labels_fine[fine_valid_inds,:]))
data = np.hstack((train_data_fine[:,:num_features_fine], np.reshape(train_data_fine[:, -1], [-1,1])))
train_data_fine = np.vstack((augment_data_fine, data))

print(train_data_coarse.shape, train_data_fine.shape)
# %%
print("RETRAIN MODELS")
saver = tf.train.Saver()
merged_summary_op = tf.summary.merge_all()
with tf.Session() as sess:
    summary_writer = tf.summary.FileWriter(os.path.join(config.MODEL_DIR, "MULTI_RES", "Augment"), sess.graph)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    k=0
    for i in range(config.MULTIRES_Augment_n_epochs):
        data_fine = train_data_fine[:, :-1]
        label_fine = np.reshape(train_data_fine[:, -1], [-1,1])
        data_coarse = train_data_coarse[:, :-1]
        label_coarse = np.reshape(train_data_coarse[:, -1], [-1,1])
        feed_dict = {X_fine:data_fine, Y_fine:label_fine, X_coarse:data_coarse, Y_coarse:label_coarse}
        summary_str, _, _, loss_fine_epoch, loss_coarse_epoch = sess.run([merged_summary_op, optimizer_coarse, optimizer_fine, loss_fine, loss_coarse], feed_dict=feed_dict)
        summary_writer.add_summary(summary_str, global_step=global_step.eval())
        if not i%100:
            print('Epoch:{0} Loss_fine:{1:.4f} Loss_coarse:{2:.4f}'.format(i, loss_fine_epoch, loss_coarse_epoch))
        k = k+1
    summary_writer.close()
    save_path = saver.save(sess, os.path.join(config.MODEL_DIR, "MULTI_RES", "Augment", "model.ckpt"))
