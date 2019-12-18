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
    n_epoch = tf.placeholder(tf.float32, shape=())
    X_fine = tf.placeholder(tf.int32, [None, num_features_fine], name="fine_res_inputs")
    Y_fine = tf.placeholder(tf.float32, [None, 1], name="fine_res_labels")
    X_coarse = tf.placeholder(tf.int32, [None, num_features_coarse], name="corase_res_inputs")
    Y_coarse = tf.placeholder(tf.float32, [None, 1], name="coarse_res_labels")
    X_fine_consistency= tf.placeholder(tf.int32, [config.max_sentences*config.MULTIRES_Attention_batch_consistency, num_features_fine], name="fine_res_consistency_inputs")
    X_coarse_consistency = tf.placeholder(tf.int32, [config.MULTIRES_Attention_batch_consistency, num_features_coarse], name="coarse_res_consistency_inputs")
    tf_embedding_placeholder = tf.placeholder(tf.float32, shape=[400001, 100])

with tf.variable_scope("Variables", reuse=tf.AUTO_REUSE):
    W_fine = tf.get_variable("Weights_layer_1_fine", [64, 1], initializer=tf.contrib.layers.xavier_initializer())
    b_fine = tf.get_variable("Biases_layer_1_fine", [1], initializer=tf.zeros_initializer())
    W_coarse = tf.get_variable("Weights_layer_1_coarse", [64, 1], initializer=tf.contrib.layers.xavier_initializer())
    b_coarse = tf.get_variable("Biases_layer_1_coarse", [1], initializer=tf.zeros_initializer())
    W_attention = tf.get_variable("Weights_attention", [128, 1], initializer=tf.contrib.layers.xavier_initializer())

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

Z_coarse_consistency, _ = tf.nn.dynamic_rnn(lstm_cell_coarse, tf.nn.embedding_lookup(params=tf_embedding, ids=X_coarse_consistency), dtype=tf.float32)
Z_coarse_consistency = Z_coarse_consistency[:,-1,:]
Z_fine_consistency, _ = tf.nn.dynamic_rnn(lstm_cell_fine, tf.nn.embedding_lookup(params=tf_embedding, ids=X_fine_consistency), dtype=tf.float32)
Z_fine_consistency = Z_fine_consistency[:,-1,:]
ind = list(itertools.chain.from_iterable(itertools.repeat(x, config.max_sentences) for x in range(config.MULTIRES_Attention_batch_consistency)))
Z_concat_consistency = tf.concat([tf.gather(Z_coarse_consistency, ind), Z_fine_consistency], axis=-1)
Z_attention_consistency = []
for i in range(config.MULTIRES_Attention_batch_consistency):
    score = tf.matmul(tf.nn.tanh(Z_concat_consistency[i*config.max_sentences:(i+1)*config.max_sentences]), W_attention)
    attention_weights = tf.nn.softmax(score, axis=0)
    context = tf.divide(tf.matmul(tf.transpose(attention_weights), Z_fine_consistency[i*config.max_sentences:(i+1)*config.max_sentences]), tf.reduce_sum(attention_weights))
    Z_attention_consistency.append(context)
Z_attention_consistency = tf.reshape(tf.convert_to_tensor(Z_attention_consistency), (config.MULTIRES_Attention_batch_consistency, Z_fine_consistency.shape[1]))
Z_coarse_consistency = tf.nn.sigmoid(tf.add(tf.matmul(Z_coarse_consistency, W_coarse, name="multiply_weights"), b_coarse, name="add_bias"))
Z_fine_consistency = tf.nn.sigmoid(tf.add(tf.matmul(Z_attention_consistency, W_fine, name="multiply_weights"), b_fine, name="add_bias"))

with tf.name_scope("loss_function"):
    switch = tf.minimum(tf.maximum(n_epoch-tf.constant(2000.0),0.0), 1.0)
    loss_fine = tf.reduce_mean(tf.losses.mean_squared_error(predictions=Z_fine, labels=Y_fine))
    loss_coarse = tf.reduce_mean(tf.losses.mean_squared_error(predictions=Z_coarse, labels=Y_coarse))
    loss_consistency = tf.reduce_mean(tf.squared_difference(Z_coarse_consistency, Z_fine_consistency))
    loss = (switch*config.MULTIRES_Attention_reg_param_1*loss_fine + config.MULTIRES_Attention_reg_param_2*loss_coarse + switch*config.MULTIRES_Attention_reg_param_3*loss_consistency)/(switch*config.MULTIRES_Attention_reg_param_1+config.MULTIRES_Attention_reg_param_2+switch*config.MULTIRES_Attention_reg_param_3)
tf.summary.scalar('loss', loss)

global_step = tf.Variable(0, name='global_step', trainable=False)
with tf.variable_scope("optimizer", reuse=tf.AUTO_REUSE):
    optimizer = tf.train.AdamOptimizer(config.MULTIRES_Attention_learning_rate).minimize(loss, global_step)
# %%
print("TRAIN MODEL")
saver = tf.train.Saver()
merged_summary_op = tf.summary.merge_all()
with tf.Session() as sess:
    summary_writer = tf.summary.FileWriter(os.path.join(config.MODEL_DIR, "MULTI_RES", "Attention"), sess.graph)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    sess.run(tf_embedding_init, feed_dict={tf_embedding_placeholder: index_to_embedding})
    k=0
    for i in range(config.MULTIRES_Attention_n_epochs):
        data_fine = train_data_fine[:, :num_features_fine]
        label_fine = np.reshape(train_data_fine[:, -1], [-1,1])
        data_coarse = train_data_coarse[:, :num_features_coarse]
        label_coarse = np.reshape(train_data_coarse[:, -1], [-1,1])
        if (k+1)*config.MULTIRES_Attention_batch_consistency>len(consistency_data_coarse):
            k = 0
        consistency_coarse = consistency_data_coarse[k*config.MULTIRES_Attention_batch_consistency:(k+1)*config.MULTIRES_Attention_batch_consistency, :]
        consistency_fine = consistency_data_fine[(k*config.max_sentences)*config.MULTIRES_Attention_batch_consistency:((k+1)*config.max_sentences)*config.MULTIRES_Attention_batch_consistency, :]
        feed_dict = {X_fine:data_fine, Y_fine:label_fine, X_coarse:data_coarse, Y_coarse:label_coarse, X_fine_consistency:consistency_fine, X_coarse_consistency:consistency_coarse, n_epoch:i}
        summary_str, _, loss_epoch, loss1, loss2, loss3, see_switch = sess.run([merged_summary_op, optimizer, loss, loss_fine, loss_coarse, loss_consistency, switch], feed_dict=feed_dict)
        summary_writer.add_summary(summary_str, global_step=global_step.eval())
        if not i%100:
            print('Epoch:{0} Loss:{1:.6f} loss1:{2:.6f} loss2:{3:.6f} loss3:{4:.6f} switch:{5}'.format(i, loss_epoch, loss1, loss2, loss3, see_switch))
        k = k+1
    summary_writer.close()
    save_path = saver.save(sess, os.path.join(config.MODEL_DIR, "MULTI_RES", "Attention", "model.ckpt"))
