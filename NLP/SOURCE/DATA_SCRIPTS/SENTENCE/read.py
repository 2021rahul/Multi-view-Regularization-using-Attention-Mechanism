#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 14:50:56 2019

@author: ghosh128
"""

import sys
sys.path.append("../../")
import os
import numpy as np
import random
import re
import config
import nltk
if not os.path.exists(config.SENTENCE_DIR):
    os.makedirs(config.SENTENCE_DIR)
# %%
print("SENTENCES")
with open(os.path.join(config.SENTENCE_DIR, "imdb_labelled.txt"), "r", encoding="utf-8") as f:
    doc = f.readlines()

index_0 = []
index_1 = []
for i,line in enumerate(doc):
    sentence, score = line.strip().split("\t")
    if int(score):
        index_1.append(i)
    else:
        index_0.append(i)
index_0 = np.array(index_0)
index = random.sample(range(len(index_0)), len(index_0))
index_0 = index_0[index]
index_1 = np.array(index_1)
index = random.sample(range(len(index_1)), len(index_1))
index_1 = index_1[index]

data_0 = []
for i in index_0:
    sentence, score = doc[i].strip().split("\t")
    sentence = sentence.replace("<br />", " ")
    sentence = re.sub('[^a-zA-Z]', ' ', sentence)
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)
    sentence = re.sub(r'\s+', ' ', sentence)
    sentence = sentence.lower()
    sentence = sentence.strip()
    num_words = len(sentence.split())
    if num_words>=10 and num_words<=config.max_sentence_length:
        if num_words < config.max_sentence_length:
            sentence = sentence + " " + " ".join((config.max_sentence_length-num_words)*["UNK"])
        data_0.append(sentence)

data_1 = []
for i in index_1:
    sentence, score = doc[i].strip().split("\t")
    sentence = sentence.replace("<br />", " ")
    sentence = re.sub('[^a-zA-Z]', ' ', sentence)
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)
    sentence = re.sub(r'\s+', ' ', sentence)
    sentence = sentence.lower()
    sentence = sentence.strip()
    num_words = len(sentence.split())
    if num_words>=10 and num_words<=config.max_sentence_length:
        if num_words < config.max_sentence_length:
            sentence = sentence + " " + " ".join((config.max_sentence_length-num_words)*["UNK"])
        data_1.append(sentence)

index_0 = random.sample(range(len(data_0)), len(data_0))
train_index_0 = index_0[:config.sentence_num_train]
test_index_0 = index_0[config.sentence_num_train:]
index_1 = random.sample(range(len(data_1)), len(data_1))
train_index_1 = index_1[:config.sentence_num_train]
test_index_1 = index_1[config.sentence_num_train:]

train_data_0 = []
train_label_0 = []
for i in train_index_0:
    train_data_0.append(data_0[i])
    train_label_0.append(0)

train_data_1 = []
train_label_1 = []
for i in train_index_1:
    train_data_1.append(data_1[i])
    train_label_1.append(1)

train_data = train_data_0 + train_data_1
train_label = train_label_0 + train_label_1

with open(os.path.join(config.SENTENCE_DIR, 'train_data.txt'), 'w') as f:
    for line in train_data:
        f.write("%s\n" % line)

with open(os.path.join(config.SENTENCE_DIR, 'train_label.txt'), 'w') as f:
    for line in train_label:
        f.write("%s\n" % line)

test_data_0 = []
test_label_0 = []
for i in test_index_0:
    test_data_0.append(data_0[i])
    test_label_0.append(0)

test_data_1 = []
test_label_1 = []
for i in test_index_1:
    test_data_1.append(data_1[i])
    test_label_1.append(1)

test_data = test_data_0 + test_data_1
test_label = test_label_0 + test_label_1

with open(os.path.join(config.SENTENCE_DIR, 'test_data.txt'), 'w') as f:
    for line in test_data:
        f.write("%s\n" % line)

with open(os.path.join(config.SENTENCE_DIR, 'test_label.txt'), 'w') as f:
    for line in test_label:
        f.write("%s\n" % line)
