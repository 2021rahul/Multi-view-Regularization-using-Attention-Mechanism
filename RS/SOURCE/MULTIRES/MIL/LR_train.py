#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 15:04:27 2019

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
    n_epoch = tf.placeholder(tf.float32, shape=())
    X_fine = tf.placeholder(tf.float32, [None, num_features_fine], name="fine_res_inputs")
    Y_fine = tf.placeholder(tf.float32, [None, 1], name="fine_res_labels")
    X_coarse = tf.placeholder(tf.float32, [None, num_features_coarse], name="corase_res_inputs")
    Y_coarse = tf.placeholder(tf.float32, [None, 1], name="coarse_res_labels")
    X_fine_consistency= tf.placeholder(tf.float32, [9*config.MULTIRES_MIL_batch_consistency, num_features_fine], name="fine_res_consistency_inputs")
    X_coarse_consistency = tf.placeholder(tf.float32, [config.MULTIRES_MIL_batch_consistency, num_features_coarse], name="coarse_res_consistency_inputs")

with tf.variable_scope("Variables", reuse=tf.AUTO_REUSE):
    W_fine = tf.get_variable("Weights_fine", [12, 1], initializer=tf.contrib.layers.xavier_initializer())
    b_fine = tf.get_variable("Biases_fine", [1], initializer=tf.zeros_initializer())
    W_coarse = tf.get_variable("Weights_coarse", [11, 1], initializer=tf.contrib.layers.xavier_initializer())
    b_coarse = tf.get_variable("Biases_coarse", [1], initializer=tf.zeros_initializer())

Z_fine = tf.nn.sigmoid(tf.add(tf.matmul(X_fine, W_fine, name="multiply_weights"), b_fine, name="add_bias"))
Z_coarse = tf.nn.sigmoid(tf.add(tf.matmul(X_coarse, W_coarse, name="multiply_weights"), b_coarse, name="add_bias"))
Z_fine_consistency = tf.nn.sigmoid(tf.add(tf.matmul(X_fine_consistency, W_fine, name="multiply_weights"), b_fine, name="add_bias"))
Z_coarse_consistency = tf.nn.sigmoid(tf.add(tf.matmul(X_coarse_consistency, W_coarse, name="multiply_weights"), b_coarse, name="add_bias"))

coarse_Z_fine_consistency = []
for i in range(config.MULTIRES_MIL_batch_consistency):
    batch_Z_fine_consistency = Z_fine_consistency[i*9:(i+1)*9]
    softmax_batch_Z_fine_consistency = tf.pow(config.MULTIRES_MIL_x, batch_Z_fine_consistency)/tf.reduce_sum(tf.pow(config.MULTIRES_MIL_x, batch_Z_fine_consistency))
    coarse_Z_fine_consistency.append(tf.matmul(softmax_batch_Z_fine_consistency, batch_Z_fine_consistency, transpose_a=True, transpose_b=False))
    # batch_Z_fine_consistency = tf.reshape(Z_fine_consistency[i*9:(i+1)*9], (-1,))
    # softmax_batch_Z_fine_consistency = tf.reshape(tf.nn.softmax(batch_Z_fine_consistency, axis=0), (-1,))
    # coarse_Z_fine_consistency.append(tf.tensordot(softmax_batch_Z_fine_consistency, batch_Z_fine_consistency, axes=1))
coarse_Z_fine_consistency = tf.reshape(tf.convert_to_tensor(coarse_Z_fine_consistency), (config.MULTIRES_MIL_batch_consistency, 1))

with tf.name_scope("loss_function"):
    switch = tf.minimum(tf.maximum(n_epoch-tf.constant(0.0),0.0), 1.0)
    loss_fine = tf.reduce_mean(tf.losses.mean_squared_error(predictions=Z_fine, labels=Y_fine))
    loss_coarse = tf.reduce_mean(tf.losses.mean_squared_error(predictions=Z_coarse, labels=Y_coarse))
    loss_consistency = tf.reduce_mean(tf.squared_difference(coarse_Z_fine_consistency, Z_coarse_consistency))
    loss = (switch*config.MULTIRES_MIL_reg_param_1*loss_fine + config.MULTIRES_MIL_reg_param_2*loss_coarse + switch*config.MULTIRES_MIL_reg_param_3*loss_consistency)/(switch*config.MULTIRES_MIL_reg_param_1+config.MULTIRES_MIL_reg_param_2+switch*config.MULTIRES_MIL_reg_param_3)
tf.summary.scalar('loss', loss)

global_step = tf.Variable(0, name='global_step', trainable=False)
with tf.variable_scope("optimizer", reuse=tf.AUTO_REUSE):
    optimizer = tf.train.AdamOptimizer(config.MULTIRES_MIL_learning_rate).minimize(loss, global_step)
# %%
print("TRAIN MODEL")
saver = tf.train.Saver()
merged_summary_op = tf.summary.merge_all()
with tf.Session() as sess:
    summary_writer = tf.summary.FileWriter(os.path.join(config.MODEL_DIR, "MULTI_RES", "MIL"), sess.graph)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    k=0
    for i in range(config.MULTIRES_MIL_n_epochs):
        data_fine = train_data_fine[:, :num_features_fine]
        label_fine = np.reshape(train_data_fine[:, -1], [-1,1])
        data_coarse = train_data_coarse[:, :num_features_coarse]
        label_coarse = np.reshape(train_data_coarse[:, -1], [-1,1])
        if k*config.MULTIRES_MIL_batch_consistency>len(consistency_data_coarse) or (k+1)*config.MULTIRES_MIL_batch_consistency>len(consistency_data_coarse):
            k = 0
        consistency_coarse = consistency_data_coarse[k*config.MULTIRES_MIL_batch_consistency:(k+1)*config.MULTIRES_MIL_batch_consistency, :]
        consistency_fine = consistency_data_fine[(k*9)*config.MULTIRES_MIL_batch_consistency:((k+1)*9)*config.MULTIRES_MIL_batch_consistency, :]
        feed_dict = {X_fine:data_fine, Y_fine:label_fine, X_coarse:data_coarse, Y_coarse:label_coarse, X_fine_consistency:consistency_fine, X_coarse_consistency:consistency_coarse, n_epoch:i}
        summary_str, _, loss_epoch, loss1, loss2, loss3, see_switch = sess.run([merged_summary_op, optimizer, loss, loss_fine, loss_coarse, loss_consistency, switch], feed_dict=feed_dict)
        summary_writer.add_summary(summary_str, global_step=global_step.eval())
        if not i%100:
            print('Epoch:{0} Loss:{1:.4f} loss1:{2:.4f} loss2:{3:.4f} loss3:{4:.4f} switch:{5}'.format(i, loss_epoch, loss1, loss2, loss3, see_switch))
        k = k+1
    summary_writer.close()
    save_path = saver.save(sess, os.path.join(config.MODEL_DIR, "MULTI_RES", "MIL", "LR_model.ckpt"))
