#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 10:42:06 2019

@author: ghosh128
"""

import sys
sys.path.append("../../")
import os
import numpy as np
import pandas as pd
import random
import re
import config
import nltk
import itertools
import pickle
import tensorflow as tf
from collections import defaultdict
# %%
print("LOAD DATA")
files_1 = os.listdir(os.path.join(config.REVIEW_DIR, "pos"))
files_1 = [file for file in files_1 if file.endswith(".txt")]
index_1 = random.sample(range(len(files_1)), len(files_1))
files_0 = os.listdir(os.path.join(config.REVIEW_DIR, "neg"))
files_0 = [file for file in files_0 if file.endswith(".txt")]
index_0 = random.sample(range(len(files_0)), len(files_0))
# %%
print("LOAD EMBEDDING DATA")
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
# # %%
num_features_fine = config.max_sentence_length
num_features_coarse = config.max_review_length

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

visualize_att_weights = []

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
    visualize_att_weights.append(attention_weights)
    context = tf.divide(tf.matmul(tf.transpose(attention_weights), Z_fine_consistency[i*config.max_sentences:(i+1)*config.max_sentences]), tf.reduce_sum(attention_weights))
    Z_attention_consistency.append(context)
Z_attention_consistency = tf.reshape(tf.convert_to_tensor(Z_attention_consistency), (config.MULTIRES_Attention_batch_consistency, Z_fine_consistency.shape[1]))
Z_coarse_consistency = tf.nn.sigmoid(tf.add(tf.matmul(Z_coarse_consistency, W_coarse, name="multiply_weights"), b_coarse, name="add_bias"))
Z_fine_consistency = tf.nn.sigmoid(tf.add(tf.matmul(Z_attention_consistency, W_fine, name="multiply_weights"), b_fine, name="add_bias"))

visualize_att_weights = tf.reshape(tf.convert_to_tensor(visualize_att_weights), (config.MULTIRES_Attention_batch_consistency, -1))
# %%
print("VISUALIZE MODEL")

actual_lines = []
data_1_coarse = []
data_1_fine = []
index=0
count = 0
while count < config.MULTIRES_Attention_batch_consistency:
    with open(os.path.join(config.REVIEW_DIR, "pos", files_1[index_1[index]]), "r", encoding="utf-8") as f:
        doc = f.readlines()
    text = doc[0].strip().replace("<br />", " ").lower()
    lines = nltk.sent_tokenize(text)
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = re.sub(r"\s+[a-zA-Z]\s+", ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()

    num_words_review = len(text.split())
    num_lines_review = len(lines)
    num_words_line = []
    clean_lines = []
    for line in lines:
        line = re.sub('[^a-zA-Z]', ' ', line)
        line = re.sub(r"\s+[a-zA-Z]\s+", ' ', line)
        line = re.sub(r'\s+', ' ', line)
        line = line.lower()
        line = line.strip()
        num_words_line.append(len(line.split()))
        clean_lines.append(line)

    if num_words_review<=config.max_review_length and num_lines_review>=4 and num_lines_review<=config.max_sentences and all(i>=5 and i<=config.max_sentence_length for i in num_words_line):
        actual_lines = actual_lines + lines
        if num_words_review<config.max_review_length:
            text = text + " " + " ".join((200-num_words_review)*["UNK"])
        data_1_coarse.append(text)
        for i,line in enumerate(clean_lines):
            if len(line.split()) < config.max_sentence_length:
                line = line + " " + " ".join((config.max_sentence_length-len(line.split()))*["UNK"])
                clean_lines[i] = line
        while len(clean_lines) < config.max_sentences:
            clean_lines.append(" ".join(config.max_sentence_length *["UNK"]))
            actual_lines.append(" ".join(config.max_sentence_length *["UNK"]))
        data_1_fine = data_1_fine+clean_lines
        count += 1
    index = index+1

visualize_data_coarse = []
for i in range(len(data_1_coarse)):
    word_indexes = [word_to_index[w] for w in data_1_coarse[i].split()]
    visualize_data_coarse.append(word_indexes)

visualize_data_fine = []
for i in range(len(data_1_fine)):
    word_indexes = [word_to_index[w] for w in data_1_fine[i].split()]
    visualize_data_fine.append(word_indexes)

saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, os.path.join(config.MODEL_DIR, "MULTI_RES", "Attention", "model.ckpt"))
    sess.run(tf_embedding_init, feed_dict={tf_embedding_placeholder: index_to_embedding})
    feed_dict = {X_fine_consistency:visualize_data_fine, X_coarse_consistency:visualize_data_coarse}
    attention_weights_arr = sess.run([visualize_att_weights], feed_dict=feed_dict)
    attention_weights_arr = np.reshape(np.array(attention_weights_arr), (config.MULTIRES_Attention_batch_consistency,10))

col_names =  ['Lines', 'Weights']
data = np.reshape(np.array(actual_lines),(-1,1))
weights = np.reshape(attention_weights_arr, (-1,1))
print(data.shape)
print(weights.shape)

my_df  = pd.DataFrame(data=np.hstack((data,weights)), columns=['Lines', 'Weights'])
pickle.dump(my_df, open("visualize_pos.pkl", "wb"))

with open("visualize_pos.txt", "w") as f:
    for i in range(len(actual_lines)):
        f.write(actual_lines[i]+" : "+str(attention_weights_arr[i//10,i%10])+"\n")
    f.write("\n")
    f.write("###################################################################\n")

print("VISUALIZE MODEL")

actual_lines = []
data_0_coarse = []
data_0_fine = []
index=0
count = 0
while count < config.MULTIRES_Attention_batch_consistency:
    with open(os.path.join(config.REVIEW_DIR, "neg", files_0[index_0[index]]), "r", encoding="utf-8") as f:
        doc = f.readlines()
    text = doc[0].strip().replace("<br />", " ").lower()
    lines = nltk.sent_tokenize(text)
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = re.sub(r"\s+[a-zA-Z]\s+", ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()

    num_words_review = len(text.split())
    num_lines_review = len(lines)
    num_words_line = []
    clean_lines = []
    for line in lines:
        line = re.sub('[^a-zA-Z]', ' ', line)
        line = re.sub(r"\s+[a-zA-Z]\s+", ' ', line)
        line = re.sub(r'\s+', ' ', line)
        line = line.lower()
        line = line.strip()
        num_words_line.append(len(line.split()))
        clean_lines.append(line)

    if num_words_review<=config.max_review_length and num_lines_review>=4 and num_lines_review<=config.max_sentences and all(i>=5 and i<=config.max_sentence_length for i in num_words_line):
        actual_lines = actual_lines + lines
        if num_words_review<config.max_review_length:
            text = text + " " + " ".join((200-num_words_review)*["UNK"])
        data_0_coarse.append(text)
        for i,line in enumerate(clean_lines):
            if len(line.split()) < config.max_sentence_length:
                line = line + " " + " ".join((config.max_sentence_length-len(line.split()))*["UNK"])
                clean_lines[i] = line
        while len(clean_lines) < config.max_sentences:
            clean_lines.append(" ".join(config.max_sentence_length *["UNK"]))
            actual_lines.append(" ".join(config.max_sentence_length *["UNK"]))
        data_0_fine = data_0_fine+clean_lines
        count += 1
    index = index+1

visualize_data_coarse = []
for i in range(len(data_0_coarse)):
    word_indexes = [word_to_index[w] for w in data_0_coarse[i].split()]
    visualize_data_coarse.append(word_indexes)

visualize_data_fine = []
for i in range(len(data_0_fine)):
    word_indexes = [word_to_index[w] for w in data_0_fine[i].split()]
    visualize_data_fine.append(word_indexes)

print("VISUALIZE MODEL")
saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, os.path.join(config.MODEL_DIR, "MULTI_RES", "Attention", "model.ckpt"))
    sess.run(tf_embedding_init, feed_dict={tf_embedding_placeholder: index_to_embedding})
    feed_dict = {X_fine_consistency:visualize_data_fine, X_coarse_consistency:visualize_data_coarse}
    attention_weights_arr = sess.run([visualize_att_weights], feed_dict=feed_dict)
    attention_weights_arr = np.reshape(np.array(attention_weights_arr), (config.MULTIRES_Attention_batch_consistency,10))

col_names =  ['Lines', 'Weights']
data = np.reshape(np.array(actual_lines),(-1,1))
weights = np.reshape(attention_weights_arr, (-1,1))
print(data.shape)
print(weights.shape)

my_df  = pd.DataFrame(data=np.hstack((data,weights)), columns=['Lines', 'Weights'])
pickle.dump(my_df, open("visualize_neg.pkl", "wb"))

with open("visualize_neg.txt", "w") as f:
    for i in range(len(actual_lines)):
        f.write(actual_lines[i]+" : "+str(attention_weights_arr[i//10,i%10])+"\n")
    f.write("\n")
    f.write("###################################################################\n")
