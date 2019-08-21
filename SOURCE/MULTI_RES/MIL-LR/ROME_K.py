#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 15:12:22 2019

@author: ghosh128
"""

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
tf.set_random_seed(1)
# %%
data_dir = "../../DATA/ROME/NUMPY/"
train_data_fine = np.load(os.path.join(data_dir, "sentinel_train_data.npy"))
train_data_coarse = np.load(os.path.join(data_dir, "landsat_train_data.npy"))
test_data_fine = np.load(os.path.join(data_dir, "sentinel_test_data.npy"))
test_data_coarse = np.load(os.path.join(data_dir, "landsat_test_data.npy"))
consistency_data_fine = np.load(os.path.join(data_dir, "sentinel_consistency_data.npy"))
consistency_data_coarse = np.load(os.path.join(data_dir, "landsat_consistency_data.npy"))
# %%
learning_rate = 0.001
n_epochs = 10000
x = 1000.0
s = 1000.0
K = 0.5
lambda1 = 1.0
lambda2 = 1.0
batch_consistency = 5000

tf.reset_default_graph()
with tf.name_scope('data'):
    X_fine = tf.placeholder(tf.float32, [None, 12], name="fine_res_inputs")
    Y_fine = tf.placeholder(tf.float32, [None, 1], name="fine_res_labels")
    X_coarse = tf.placeholder(tf.float32, [None, 11], name="corase_res_inputs")
    Y_coarse = tf.placeholder(tf.float32, [None, 1], name="coarse_res_labels")
    X_fine_consistency= tf.placeholder(tf.float32, [9*batch_consistency, 12], name="fine_res_consistency_inputs")
    X_coarse_consistency = tf.placeholder(tf.float32, [batch_consistency, 11], name="coarse_res_consistency_inputs")

with tf.variable_scope("Variables", reuse=tf.AUTO_REUSE):
    W_fine = tf.get_variable("Weights_fine", [12, 1], initializer=tf.contrib.layers.xavier_initializer())
    b_fine = tf.get_variable("Biases_fine", [1], initializer=tf.zeros_initializer())
    W_coarse = tf.get_variable("Weights_coarse", [11, 1], initializer=tf.contrib.layers.xavier_initializer())
    b_coarse = tf.get_variable("Biases_coarse", [1], initializer=tf.zeros_initializer())

Z_fine = tf.matmul(X_fine, W_fine, name="multiply_weights")
Z_fine = tf.add(Z_fine, b_fine, name="add_bias")
pred_fine = tf.nn.sigmoid(Z_fine)

Z_coarse = tf.matmul(X_coarse, W_coarse, name="multiply_weights")
Z_coarse = tf.add(Z_coarse, b_coarse, name="add_bias")
pred_coarse = tf.nn.sigmoid(Z_coarse)

Z_fine_consistency = tf.matmul(X_fine_consistency, W_fine, name="multiply_weights")
Z_fine_consistency = tf.add(Z_fine_consistency, b_fine, name="add_bias")
pred_fine_consistency = tf.nn.sigmoid(Z_fine_consistency)

Z_coarse_consistency = tf.matmul(X_coarse_consistency, W_coarse, name="multiply_weights")
Z_coarse_consistency = tf.add(Z_coarse_consistency, b_coarse, name="add_bias")
pred_coarse_consistency = tf.nn.sigmoid(Z_coarse_consistency)

coarse_pred_fine_consistency = []
for i in range(batch_consistency):
    coarse_pred_fine_consistency.append(tf.divide(tf.reduce_sum(pred_fine_consistency[i*9:(i+1)*9]), K))
coarse_pred_fine_consistency = tf.reshape(tf.convert_to_tensor(coarse_pred_fine_consistency), (batch_consistency, 1))

with tf.name_scope("loss_function"):
    loss_fine = tf.reduce_mean(tf.losses.mean_squared_error(predictions=pred_fine, labels=Y_fine))
    loss_coarse = tf.reduce_mean(tf.losses.mean_squared_error(predictions=pred_coarse, labels=Y_coarse))
    loss_consistency = tf.reduce_mean(tf.subtract(1.0, tf.nn.sigmoid(tf.multiply(s*(9/K), tf.subtract(coarse_pred_fine_consistency, pred_coarse_consistency)))))
    loss = (loss_fine + lambda1*loss_coarse + lambda2*loss_consistency)
tf.summary.scalar('loss', loss)

global_step = tf.Variable(0, name='global_step', trainable=False)

with tf.variable_scope("optimizer", reuse=tf.AUTO_REUSE):
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step)
# %%
# TRAIN MODEL
saver = tf.train.Saver()
merged_summary_op = tf.summary.merge_all()
with tf.Session() as sess:
    summary_writer = tf.summary.FileWriter("../../MODEL/SENTINEL_LANDSAT-LR", sess.graph)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    k=0
    for i in range(n_epochs):
        data_fine = train_data_fine[:, :-1]
        label_fine = np.reshape(train_data_fine[:, -1], [-1,1])
        data_coarse = train_data_coarse[:, :-1]
        label_coarse = np.reshape(train_data_coarse[:, -1], [-1,1])
        if k*batch_consistency>len(consistency_data_coarse) or (k+1)*batch_consistency>len(consistency_data_coarse):
            k = 0
        consistency_coarse = consistency_data_coarse[k*batch_consistency:(k+1)*batch_consistency, :]
        consistency_fine = consistency_data_fine[(k*9)*batch_consistency:((k+1)*9)*batch_consistency, :]
        feed_dict = {X_fine: data_fine, Y_fine: label_fine, X_coarse: data_coarse, Y_coarse: label_coarse, X_fine_consistency: consistency_fine, X_coarse_consistency: consistency_coarse}
        summary_str, _, loss_epoch = sess.run([merged_summary_op, optimizer, loss], feed_dict=feed_dict)
        summary_writer.add_summary(summary_str, global_step=global_step.eval())
        if not i%1000:
            print('Average loss epoch {0}: {1}'.format(i, loss_epoch))
        k = k+1
    summary_writer.close()
    save_path = saver.save(sess, "../../MODEL/SENTINEL_LANDSAT-LR/ROME.CKPT")
# %%
# TEST MODEL ON SENTINEL TRAIN DATA
with tf.Session() as sess:
    saver.restore(sess, "../../MODEL/SENTINEL_LANDSAT-LR/ROME.CKPT")
    data = train_data_fine[:, :-1]
    feed_dict = {X_fine: data}
    preds = sess.run(pred_fine, feed_dict=feed_dict)
    pred_labels = np.zeros(preds.shape)
    pred_labels[preds > 0.5] = 1
    pred_labels[preds < 0.5] = 0
    actual_labels = np.reshape(train_data_fine[:, -1], (-1, 1))
    print("Accuracy:", len(np.where([pred_labels == actual_labels])[1])/int(len(actual_labels)))
    print("f1_score:", f1_score(actual_labels, pred_labels))
plt.hist(preds[actual_labels==0], color="red")
plt.hist(preds[actual_labels==1], color="green")
plt.show()
# %%
# TEST MODEL ON LANDSAT TRAIN DATA
with tf.Session() as sess:
    saver.restore(sess, "../../MODEL/SENTINEL_LANDSAT-LR/ROME.CKPT")
    data = train_data_coarse[:, :-1]
    feed_dict = {X_coarse: data}
    preds = sess.run(pred_coarse, feed_dict=feed_dict)
    pred_labels = np.zeros(preds.shape)
    pred_labels[preds > 0.5] = 1
    pred_labels[preds < 0.5] = 0
    actual_labels = np.reshape(train_data_coarse[:, -1], (-1, 1))
    print("Accuracy:", len(np.where([pred_labels == actual_labels])[1])/int(len(actual_labels)))
    print("f1_score:", f1_score(actual_labels, pred_labels))
plt.hist(preds[actual_labels==0], color="red")
plt.hist(preds[actual_labels==1], color="green")
plt.show()
#%%
# TEST MODEL ON SENTINEL TEST DATA
with tf.Session() as sess:
    saver.restore(sess, "../../MODEL/SENTINEL_LANDSAT-LR/ROME.CKPT")
    data = test_data_fine[:, :-1]
    feed_dict = {X_fine: data}
    preds = sess.run(pred_fine, feed_dict=feed_dict)
    pred_labels = np.zeros(preds.shape)
    pred_labels[preds > 0.5] = 1
    pred_labels[preds < 0.5] = 0
    actual_labels = np.reshape(test_data_fine[:, -1], (-1, 1))
    print("Accuracy:", len(np.where([pred_labels == actual_labels])[1])/int(len(actual_labels)))
    print("f1_score:", f1_score(actual_labels, pred_labels))
plt.hist(preds[actual_labels==0], color="red")
plt.hist(preds[actual_labels==1], color="green")
plt.show()
# %%
# TEST MODEL ON LANDSAT TEST DATA
with tf.Session() as sess:
    saver.restore(sess, "../../MODEL/SENTINEL_LANDSAT-LR/ROME.CKPT")
    data = test_data_coarse[:, :-1]
    feed_dict = {X_coarse: data}
    preds = sess.run(pred_coarse, feed_dict=feed_dict)
    pred_labels = np.zeros(preds.shape)
    pred_labels[preds > 0.5] = 1
    pred_labels[preds < 0.5] = 0
    actual_labels = np.reshape(test_data_coarse[:, -1], (-1, 1))
    print("Accuracy:", len(np.where([pred_labels == actual_labels])[1])/int(len(actual_labels)))
    print("f1_score:", f1_score(actual_labels, pred_labels))
plt.hist(preds[actual_labels==0], color="red")
plt.hist(preds[actual_labels==1], color="green")
plt.show()
