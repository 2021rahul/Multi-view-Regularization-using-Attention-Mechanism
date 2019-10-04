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
from collections import defaultdict
tf.set_random_seed(1)
# %%
print("LOAD DATA")
train_data_fine = np.load(os.path.join(config.NUMPY_DIR, "sentence_train_data.npy"))
train_data_coarse = np.load(os.path.join(config.NUMPY_DIR, "review_train_data.npy"))
consistency_data_fine = np.load(os.path.join(config.NUMPY_DIR, "consistency_data_fine.npy"))
consistency_data_coarse = np.load(os.path.join(config.NUMPY_DIR, "consistency_data_coarse.npy"))
num_features_fine = config.max_sentence_length
num_features_coarse = config.max_review_length

print("LOAD EMBEDDING")
word_to_index = dict()
index_to_embedding = []
with open(os.path.join(config.EMBEDDING_DIR, "glove.6B.100d.txt"), "r", encoding="utf-8") as f:
    for (i, line) in enumerate(f):
        split = line.split(' ')
        word = split[0]
        representation = split[1:]
        representation = np.array([float(val) for val in representation])
        word_to_index[word] = i
        index_to_embedding.append(representation)
_WORD_NOT_FOUND = [0.0]* len(representation)
_LAST_INDEX = i + 1
word_to_index = defaultdict(lambda: _LAST_INDEX, word_to_index)
index_to_embedding = np.array(index_to_embedding + [_WORD_NOT_FOUND])
# %%
print("BUILD MODEL")
tf.reset_default_graph()
with tf.name_scope('data'):
    X_fine = tf.placeholder(tf.int32, [None, num_features_fine], name="fine_res_inputs")
    Y_fine = tf.placeholder(tf.float32, [None, 1], name="fine_res_labels")
    X_coarse = tf.placeholder(tf.int32, [None, num_features_coarse], name="corase_res_inputs")
    Y_coarse = tf.placeholder(tf.float32, [None, 1], name="coarse_res_labels")
    tf_embedding_placeholder = tf.placeholder(tf.float32, shape=[400001, 100])

with tf.variable_scope("Variables", reuse=tf.AUTO_REUSE):
    W_fine = tf.get_variable("Weights_layer_1_fine", [64, 1], initializer=tf.contrib.layers.xavier_initializer())
    b_fine = tf.get_variable("Biases_layer_1_fine", [1], initializer=tf.zeros_initializer())
    W_coarse = tf.get_variable("Weights_layer_1_coarse", [64, 1], initializer=tf.contrib.layers.xavier_initializer())
    b_coarse = tf.get_variable("Biases_layer_1_coarse", [1], initializer=tf.zeros_initializer())

tf_embedding = tf.Variable(tf.constant(0.0, shape=[400001, 100]), trainable=False, name="Embedding")
tf_embedding_init = tf_embedding.assign(tf_embedding_placeholder)

with tf.variable_scope("coarse", reuse=tf.AUTO_REUSE):
    lstm_cell_coarse = tf.nn.rnn_cell.LSTMCell(64, forget_bias=1.0)
    state_series_coarse, current_state_coarse = tf.nn.dynamic_rnn(lstm_cell_coarse, tf.nn.embedding_lookup(params=tf_embedding, ids=X_coarse), dtype=tf.float32)
Z_coarse = tf.nn.sigmoid(tf.add(tf.matmul(state_series_coarse[:,-1,:], W_coarse, name="multiply_weights"), b_coarse, name="add_bias"))

with tf.variable_scope("fine", reuse=tf.AUTO_REUSE):
    lstm_cell_fine = tf.nn.rnn_cell.LSTMCell(64, forget_bias=1.0)
    state_series_fine, current_state_fine = tf.nn.dynamic_rnn(lstm_cell_fine, tf.nn.embedding_lookup(params=tf_embedding, ids=X_fine), dtype=tf.float32)
Z_fine = tf.nn.sigmoid(tf.add(tf.matmul(state_series_fine[:,-1,:], W_fine, name="multiply_weights"), b_fine, name="add_bias"))

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
    sess.run(tf_embedding_init, feed_dict={tf_embedding_placeholder: index_to_embedding})
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
    sess.run(tf_embedding_init, feed_dict={tf_embedding_placeholder: index_to_embedding})
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
    sess.run(tf_embedding_init, feed_dict={tf_embedding_placeholder: index_to_embedding})
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
