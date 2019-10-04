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
from collections import defaultdict
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
tf.set_random_seed(1)
# %%
print("LOAD DATA")
test_data_fine = np.load(os.path.join(config.NUMPY_DIR, "sentence_test_data.npy"))
test_data_coarse = np.load(os.path.join(config.NUMPY_DIR, "review_test_data.npy"))
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
print("BUILD COARSE MODEL")
tf.reset_default_graph()
with tf.name_scope('data'):
    X_coarse = tf.placeholder(tf.int32, [None, num_features_coarse], name="corase_res_inputs")
    Y_coarse = tf.placeholder(tf.float32, [None, 1], name="coarse_res_labels")
    tf_embedding_placeholder = tf.placeholder(tf.float32, shape=[400001, 100])

with tf.variable_scope("Variables", reuse=tf.AUTO_REUSE):
    W_coarse = tf.get_variable("Weights_layer_1_coarse", [64, 1], initializer=tf.contrib.layers.xavier_initializer())
    b_coarse = tf.get_variable("Biases_layer_1_coarse", [1], initializer=tf.zeros_initializer())

tf_embedding = tf.Variable(tf.constant(0.0, shape=[400001, 100]), trainable=False, name="Embedding")
tf_embedding_init = tf_embedding.assign(tf_embedding_placeholder)

with tf.variable_scope("coarse", reuse=tf.AUTO_REUSE):
    lstm_cell_coarse = tf.nn.rnn_cell.LSTMCell(64, forget_bias=1.0)
    state_series_coarse, current_state_coarse = tf.nn.dynamic_rnn(lstm_cell_coarse, tf.nn.embedding_lookup(params=tf_embedding, ids=X_coarse), dtype=tf.float32)
Z_coarse = tf.nn.sigmoid(tf.add(tf.matmul(state_series_coarse[:,-1,:], W_coarse, name="multiply_weights"), b_coarse, name="add_bias"))
# %%
print("TEST COARSE MODEL")
saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, os.path.join(config.MODEL_DIR, "MULTI_RES", "WeaSL", "model_coarse.ckpt"))
    sess.run(tf_embedding_init, feed_dict={tf_embedding_placeholder: index_to_embedding})
    data_coarse = test_data_coarse[:,:num_features_coarse]
    feed_dict = {X_coarse: data_coarse}
    preds_coarse = sess.run(Z_coarse, feed_dict=feed_dict)

pred_labels_coarse = np.zeros(preds_coarse.shape)
pred_labels_coarse[preds_coarse > 0.5] = 1
pred_labels_coarse[preds_coarse < 0.5] = 0
labels_coarse = np.reshape(test_data_coarse[:, -1], [-1, 1])
# %%
print("BUILD FINE MODEL")
tf.reset_default_graph()
with tf.name_scope('data'):
    X_fine = tf.placeholder(tf.int32, [None, num_features_fine], name="fine_res_inputs")
    Y_fine = tf.placeholder(tf.float32, [None, 1], name="fine_res_labels")
    tf_embedding_placeholder = tf.placeholder(tf.float32, shape=[400001, 100])

with tf.variable_scope("Variables", reuse=tf.AUTO_REUSE):
    W_fine = tf.get_variable("Weights_layer_1_fine", [64, 1], initializer=tf.contrib.layers.xavier_initializer())
    b_fine = tf.get_variable("Biases_layer_1_fine", [1], initializer=tf.zeros_initializer())

tf_embedding = tf.Variable(tf.constant(0.0, shape=[400001, 100]), trainable=False, name="Embedding")
tf_embedding_init = tf_embedding.assign(tf_embedding_placeholder)

with tf.variable_scope("fine", reuse=tf.AUTO_REUSE):
    lstm_cell_fine = tf.nn.rnn_cell.LSTMCell(64, forget_bias=1.0)
    state_series_fine, current_state_fine = tf.nn.dynamic_rnn(lstm_cell_fine, tf.nn.embedding_lookup(params=tf_embedding, ids=X_fine), dtype=tf.float32)
Z_fine = tf.nn.sigmoid(tf.add(tf.matmul(state_series_fine[:,-1,:], W_fine, name="multiply_weights"), b_fine, name="add_bias"))
#%%
print("TEST FINE MODEL")
saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, os.path.join(config.MODEL_DIR, "MULTI_RES", "WeaSL", "model_fine.ckpt"))
    sess.run(tf_embedding_init, feed_dict={tf_embedding_placeholder: index_to_embedding})
    data_fine = test_data_fine[:,:num_features_fine]
    feed_dict = {X_fine: data_fine}
    preds_fine = sess.run(Z_fine, feed_dict=feed_dict)

pred_labels_fine = np.zeros(preds_fine.shape)
pred_labels_fine[preds_fine > 0.5] = 1
pred_labels_fine[preds_fine < 0.5] = 0
labels_fine = np.reshape(test_data_fine[:, -1], [-1, 1])

print("Accuracy_fine:", len(np.where([pred_labels_fine == labels_fine])[1])/int(len(labels_fine)), "Accuracy_coarse:", len(np.where([pred_labels_coarse == labels_coarse])[1])/int(len(labels_coarse)))
print("f1_score_fine:", f1_score(labels_fine, pred_labels_fine), "f1_score_coarse:", f1_score(labels_coarse, pred_labels_coarse))
plt.hist(preds_fine[labels_fine==0], color="red")
plt.hist(preds_fine[labels_fine==1], color="green")
plt.show()
plt.hist(preds_coarse[labels_coarse==0], color="red")
plt.hist(preds_coarse[labels_coarse==1], color="green")
plt.show()
