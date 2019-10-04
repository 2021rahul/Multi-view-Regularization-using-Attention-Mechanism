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
if not os.path.exists(config.REVIEW_DIR):
    os.makedirs(config.REVIEW_DIR)
# %%
print("REVIEWS")
files_1 = os.listdir(os.path.join(config.REVIEW_DIR, "pos"))
files_1 = [file for file in files_1 if file.endswith(".txt")]
index_1 = random.sample(range(len(files_1)), len(files_1))
files_0 = os.listdir(os.path.join(config.REVIEW_DIR, "neg"))
files_0 = [file for file in files_0 if file.endswith(".txt")]
index_0 = random.sample(range(len(files_0)), len(files_0))

num_words = []

data_0 = []
for i in index_1:
    with open(os.path.join(config.REVIEW_DIR, "pos", files_1[i]), "r", encoding="utf-8") as f:
        doc = f.readlines()
    if len(doc)>1:
        print(i)
    line = doc[0].strip().replace("<br />", " ")
    line = re.sub('[^a-zA-Z]', ' ', line)
    line = re.sub(r"\s+[a-zA-Z]\s+", ' ', line)
    line = re.sub(r'\s+', ' ', line)
    line = line.lower()
    line = line.strip()
    num_words = len(line.split())
    if num_words>=100 and num_words<=config.max_review_length:
        if num_words < config.max_review_length:
            line = line + " " + " ".join((config.max_review_length-num_words)*["UNK"])
        data_0.append(line)

data_1 = []
for i in index_0:
    with open(os.path.join(config.REVIEW_DIR, "neg", files_0[i]), "r", encoding="utf-8") as f:
        doc = f.readlines()
    if len(doc)>1:
        print(i)
    line = doc[0].strip().replace("<br />", " ")
    line = re.sub('[^a-zA-Z]', ' ', line)
    line = re.sub(r"\s+[a-zA-Z]\s+", ' ', line)
    line = re.sub(r'\s+', ' ', line)
    line = line.lower()
    line = line.strip()
    num_words = len(line.split())
    if num_words>=100 and num_words<=config.max_review_length:
        if num_words < config.max_review_length:
            line = line + " " + " ".join((config.max_review_length-num_words)*["UNK"])
        data_1.append(line)

index_0 = random.sample(range(len(data_0)), len(data_0))
train_index_0 = index_0[:config.review_num_train]
test_index_0 = index_0[config.review_num_train:config.review_num_train+config.review_num_test]
index_1 = random.sample(range(len(data_1)), len(data_1))
train_index_1 = index_1[:config.review_num_train]
test_index_1 = index_1[config.review_num_train:config.review_num_train+config.review_num_test]

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

with open(os.path.join(config.REVIEW_DIR, 'train_data.txt'), 'w') as f:
    for line in train_data:
        f.write("%s\n" % line)

with open(os.path.join(config.REVIEW_DIR, 'train_label.txt'), 'w') as f:
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

with open(os.path.join(config.REVIEW_DIR, 'test_data.txt'), 'w') as f:
    for line in test_data:
        f.write("%s\n" % line)

with open(os.path.join(config.REVIEW_DIR, 'test_label.txt'), 'w') as f:
    for line in test_label:
        f.write("%s\n" % line)
