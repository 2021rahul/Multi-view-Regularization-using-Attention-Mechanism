#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 10:12:28 2019

@author: ghosh128
"""

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
tf.set_random_seed(1)
# %%
REGION = "ROME"
data_dir = "../../DATA/"+REGION+"/NUMPY/"
num_train_samples = 200
train_data_labeled = np.load(os.path.join(data_dir, "landsat_train_data_"+str(num_train_samples)+".npy"))
train_data_unlabeled = np.load(os.path.join(data_dir, "landsat_consistency_data.npy"))
test_data = np.load(os.path.join(data_dir, "landsat_test_data.npy"))
# %%
learning_rate = 0.001
n_epochs = 10000
lam = 0.1
batch_size = 1000

tf.reset_default_graph()
with tf.name_scope('data'):
    X_labeled = tf.placeholder(tf.float32, [None, 11], name="inputs_labeled")
    X_unlabeled = tf.placeholder(tf.float32, [None, 11], name="inputs_unlabeled")
    Y = tf.placeholder(tf.float32, [None, 1], name="labels")

with tf.variable_scope("Variables", reuse=tf.AUTO_REUSE):
    W = tf.get_variable("Weights", [11, 1], initializer=tf.contrib.layers.xavier_initializer())
    b = tf.get_variable("Biases", [1], initializer=tf.zeros_initializer())

Z_labeled = tf.matmul(X_labeled, W, name="multiply_weights")
Z_labeled = tf.add(Z_labeled, b, name="add_bias")
pred_labeled = tf.nn.sigmoid(Z_labeled)

Z_unlabeled = tf.matmul(X_unlabeled, W, name="multiply_weights")
Z_unlabeled = tf.add(Z_unlabeled, b, name="add_bias")
Z_unlabeled = tf.nn.sigmoid(Z_unlabeled)

with tf.name_scope("loss_function"):
    squared_loss = tf.reduce_mean(tf.losses.mean_squared_error(predictions=pred_labeled, labels=Y))
    fx_diff = tf.square(tf.subtract(Z_unlabeled, tf.transpose(Z_unlabeled)))
    r = tf.reshape(tf.reduce_sum(X_unlabeled*X_unlabeled, 1), [-1, 1])
    x_dist = r - 2*tf.matmul(X_unlabeled, tf.transpose(X_unlabeled)) + tf.transpose(r)
    corr_loss = -tf.contrib.metrics.streaming_pearson_correlation(tf.reshape(fx_diff, [-1,1]), tf.reshape(x_dist, [-1,1]))[1]
    loss = tf.reduce_mean(squared_loss + lam*corr_loss)
tf.summary.scalar('loss', loss)

global_step = tf.Variable(0, name='global_step', trainable=False)

with tf.variable_scope("optimizer", reuse=tf.AUTO_REUSE):
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step)
# %%
# TRAIN MODEL
saver = tf.train.Saver()
merged_summary_op = tf.summary.merge_all()
with tf.Session() as sess:
    summary_writer = tf.summary.FileWriter("../../MODEL/LANDSAT/SemiSupervised_LogReg", sess.graph)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    k=0
    for i in range(n_epochs):
        data_labeled = train_data_labeled[:, :11]
        label = np.reshape(train_data_labeled[:, -1], [-1,1])

        if k*batch_size>len(train_data_unlabeled) or (k+1)*batch_size>len(train_data_unlabeled):
            k = 0
        data_unlabeled = train_data_unlabeled[(k*batch_size)%len(train_data_unlabeled):((k+1)*batch_size)%len(train_data_unlabeled), :]

        feed_dict = {X_labeled: data_labeled, X_unlabeled: data_unlabeled, Y: label}
        summary_str, _, loss_epoch = sess.run([merged_summary_op, optimizer, loss], feed_dict=feed_dict)
        summary_writer.add_summary(summary_str, global_step=global_step.eval())
        if not i%100:
            print('Average loss epoch {0}: {1}'.format(i, loss_epoch))
    summary_writer.close()
    save_path = saver.save(sess, "../../MODEL/LANDSAT/SemiSupervised_LogReg/"+REGION+"_"+str(num_train_samples)+".CKPT")
# %%
# TEST MODEL ON TRAIN DATA
with tf.Session() as sess:
    saver.restore(sess, "../../MODEL/LANDSAT/SemiSupervised_LogReg/"+REGION+"_"+str(num_train_samples)+".CKPT")
    data = train_data_labeled[:, :11]
    feed_dict = {X_labeled: data}
    preds = sess.run(pred_labeled, feed_dict=feed_dict)
    pred_labels = np.zeros(preds.shape)
    pred_labels[preds > 0.5] = 1
    pred_labels[preds < 0.5] = 0
    actual_labels = np.reshape(train_data_labeled[:, -1], (-1, 1))
    print("Accuracy:", len(np.where([pred_labels == actual_labels])[1])/int(len(actual_labels)))
    print("f1_score:", f1_score(actual_labels, pred_labels))
plt.hist(preds[actual_labels==0], color="red")
plt.hist(preds[actual_labels==1], color="green")
plt.show()
# %%
# TEST MODEL ON TEST DATA
with tf.Session() as sess:
    saver.restore(sess, "../../MODEL/LANDSAT/SemiSupervised_LogReg/"+REGION+"_"+str(num_train_samples)+".CKPT")
    data = test_data[:, :11]
    feed_dict = {X_labeled: data}
    preds = sess.run(pred_labeled, feed_dict=feed_dict)
    pred_labels = np.zeros(preds.shape)
    pred_labels[preds > 0.5] = 1
    pred_labels[preds < 0.5] = 0
    actual_labels = np.reshape(test_data[:, -1], (-1, 1))
    print("Accuracy:", len(np.where([pred_labels == actual_labels])[1])/int(len(actual_labels)))
    print("f1_score:", f1_score(actual_labels, pred_labels))
plt.hist(preds[actual_labels==0], color="red")
plt.hist(preds[actual_labels==1], color="green")
plt.show()
