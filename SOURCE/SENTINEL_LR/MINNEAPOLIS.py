#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 14:03:47 2019

@author: ghosh128
"""

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
tf.set_random_seed(1)
# %%
data_dir = "../../DATA/MINNEAPOLIS/SENTINEL/"
train_data_strong = np.load(os.path.join(data_dir, "balanced_train_data_strong_20.npy"))
train_data_weak = np.load(os.path.join(data_dir, "balanced_train_data_weak.npy"))
validate_data = np.load(os.path.join(data_dir, "balanced_validate_data.npy"))
test_data = np.load(os.path.join(data_dir, "balanced_test_data.npy"))
# %%
learning_rate = 0.01
n_epochs = 10000

tf.reset_default_graph()
with tf.name_scope('data'):
    X = tf.placeholder(tf.float32, [None, 322], name="inputs")
    Y = tf.placeholder(tf.float32, [None, 1], name="labels")

with tf.variable_scope("Variables", reuse=tf.AUTO_REUSE):
    W = tf.get_variable("Weights", [322, 1], initializer=tf.contrib.layers.xavier_initializer())
    b = tf.get_variable("Biases", [1], initializer=tf.zeros_initializer())

Z = tf.matmul(X, W, name="multiply_weights")
Z = tf.add(Z, b, name="add_bias")
pred = tf.nn.sigmoid(Z)

with tf.name_scope("loss_function"):
    loss = tf.reduce_mean(tf.losses.mean_squared_error(predictions=pred, labels=Y))
tf.summary.scalar('loss', loss)

global_step = tf.Variable(0, name='global_step', trainable=False)

with tf.variable_scope("optimizer", reuse=tf.AUTO_REUSE):
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step)
# %%
# TRAIN MODEL
saver = tf.train.Saver()
merged_summary_op = tf.summary.merge_all()
with tf.Session() as sess:
    summary_writer = tf.summary.FileWriter("../../MODEL/S-LR", sess.graph)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    for i in range(n_epochs):
        data = train_data_strong[:, 2:2+322]
        label = np.reshape(train_data_strong[:, -1], [-1,1])
        feed_dict = {X: data, Y: label}
        summary_str, _, loss_epoch = sess.run([merged_summary_op, optimizer, loss], feed_dict=feed_dict)
        summary_writer.add_summary(summary_str, global_step=global_step.eval())
        print('Average loss epoch {0}: {1}'.format(i, loss_epoch))
    summary_writer.close()
    save_path = saver.save(sess, "../../MODEL/S-LR/model_balanced.ckpt")
# %%
# TEST MODEL ON TRAIN DATA
with tf.Session() as sess:
    saver.restore(sess, "../../MODEL/S-LR/model_balanced.ckpt")
    data = train_data_weak[:, 2:2+322]
    feed_dict = {X: data}
    preds = sess.run(pred, feed_dict=feed_dict)
    pred_labels = np.zeros(preds.shape)
    pred_labels[preds > 0.5] = 1
    pred_labels[preds < 0.5] = 0
    actual_labels = np.reshape(train_data_weak[:, -1], (-1, 1))
    print("Accuracy:", len(np.where([pred_labels == actual_labels])[1])/int(len(actual_labels)))
    print("f1_score:", f1_score(actual_labels, pred_labels))
plt.hist(preds[actual_labels==0], color="red")
plt.hist(preds[actual_labels==1], color="green")
plt.show()
# %%
# TEST MODEL ON VALIDATE DATA
with tf.Session() as sess:
    saver.restore(sess, "../../MODEL/S-LR/model_balanced.ckpt")
    data = validate_data[:, 2:2+322]
    feed_dict = {X: data}
    preds = sess.run(pred, feed_dict=feed_dict)
    pred_labels = np.zeros(preds.shape)
    pred_labels[preds > 0.5] = 1
    pred_labels[preds < 0.5] = 0
    actual_labels = np.reshape(validate_data[:, -1], (-1, 1))
    print("Accuracy:", len(np.where([pred_labels == actual_labels])[1])/int(len(actual_labels)))
    print("f1_score:", f1_score(actual_labels, pred_labels))
plt.hist(preds[actual_labels==0], color="red")
plt.hist(preds[actual_labels==1], color="green")
plt.show()
# %%
# TEST MODEL ON TEST DATA
with tf.Session() as sess:
    saver.restore(sess, "../../MODEL/S-LR/model_balanced.ckpt")
    data = test_data[:, 2:2+322]
    feed_dict = {X: data}
    preds = sess.run(pred, feed_dict=feed_dict)
    pred_labels = np.zeros(preds.shape)
    pred_labels[preds > 0.5] = 1
    pred_labels[preds < 0.5] = 0
    actual_labels = np.reshape(test_data[:, -1], (-1, 1))
    print("Accuracy:", len(np.where([pred_labels == actual_labels])[1])/int(len(actual_labels)))
    print("f1_score:", f1_score(actual_labels, pred_labels))
plt.hist(preds[actual_labels==0], color="red")
plt.hist(preds[actual_labels==1], color="green")
plt.show()
