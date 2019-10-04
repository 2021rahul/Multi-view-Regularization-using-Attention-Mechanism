#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 13:50:08 2019

@author: rahul2021
"""

import sys
sys.path.append("../../")
import os
import numpy as np
import config
from collections import defaultdict

if not os.path.exists(config.NUMPY_DIR):
    os.makedirs(config.NUMPY_DIR)
#%%
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
#%%
with open(os.path.join(config.REVIEW_DIR, "consistency_data_coarse.txt"), "r", encoding="utf-8") as f_data:
    data = f_data.readlines()
train_data_coarse = []
for i in range(len(data)):
    word_indexes = [word_to_index[w] for w in data[i].split()]
    train_data_coarse.append(word_indexes)

train_data_coarse = np.array(train_data_coarse)
np.save(os.path.join(config.NUMPY_DIR, "consistency_data_coarse"), train_data_coarse)

with open(os.path.join(config.REVIEW_DIR, "consistency_data_fine.txt"), "r", encoding="utf-8") as f_data:
    data = f_data.readlines()
train_data_fine = []
for i in range(len(data)):
    word_indexes = [word_to_index[w] for w in data[i].split()]
    train_data_fine.append(word_indexes)

train_data_fine = np.array(train_data_fine)
np.save(os.path.join(config.NUMPY_DIR, "consistency_data_fine"), train_data_fine)
