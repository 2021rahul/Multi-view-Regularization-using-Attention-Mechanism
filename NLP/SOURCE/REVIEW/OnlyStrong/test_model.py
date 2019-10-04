#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 16:50:17 2019

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
test_data = np.load(os.path.join(config.NUMPY_DIR, "review_test_data.npy"))
num_features = config.max_review_length

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
    X = tf.placeholder(tf.int32, [None, config.max_review_length], name="inputs")
    Y = tf.placeholder(tf.float32, [None, 1], name="labels")
    tf_embedding_placeholder = tf.placeholder(tf.float32, shape=[400001, 100])

with tf.variable_scope("Variables", reuse=tf.AUTO_REUSE):
    W = tf.get_variable("Weights_layer", [64, 1], initializer=tf.contrib.layers.xavier_initializer())
    b = tf.get_variable("Biases_layer", [1], initializer=tf.zeros_initializer())

tf_embedding = tf.Variable(tf.constant(0.0, shape=index_to_embedding.shape), trainable=False, name="Embedding")
tf_embedding_init = tf_embedding.assign(tf_embedding_placeholder)

lstm_cell = tf.contrib.rnn.BasicLSTMCell(64, forget_bias=1.0)
state_series, current_state = tf.nn.dynamic_rnn(lstm_cell, tf.nn.embedding_lookup(params=tf_embedding, ids=X), dtype=tf.float32)
Z = tf.nn.sigmoid(tf.add(tf.matmul(state_series[:,-1,:], W, name="multiply_weights"), b, name="add_bias"))
# %%
print("TEST MODEL")
saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, os.path.join(config.MODEL_DIR, "REVIEW", "OnlyStrong", "model.ckpt"))
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
