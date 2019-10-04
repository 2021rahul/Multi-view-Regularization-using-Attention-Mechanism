#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 17:21:10 2019

@author: ghosh128
"""

import os

DATASET = "imdb"
DATA_DIR = os.path.join("../../../DATA", DATASET)
SENTENCE_DIR = os.path.join(DATA_DIR, "SENTENCE")
REVIEW_DIR = os.path.join(DATA_DIR, "REVIEW")
EMBEDDING_DIR = os.path.join("../../../DATA", "Embedding")
NUMPY_DIR = os.path.join(DATA_DIR, "NUMPY")
RESULT_DIR = os.path.join(DATA_DIR, "RESULT")
MODEL_DIR = os.path.join(DATA_DIR, "MODEL")

# DATA INFO
sentence_num_train = 50
review_num_train = 500
review_num_test = 1000
num_consistency = 1000
max_sentences = 9
max_review_length = 200
max_sentence_length = 20
embedding_dimensions = 100

# OnlyStrong
REVIEW_OnlyStrong_learning_rate = 0.0001
REVIEW_OnlyStrong_n_epochs = 2500

SENTENCE_OnlyStrong_learning_rate = 0.0001
SENTENCE_OnlyStrong_n_epochs = 2500

# SSRManifold
REVIEW_SSRManifold_learning_rate = 0.0001
REVIEW_SSRManifold_n_epochs = 2500
REVIEW_SSRManifold_reg_param = 1.0
REVIEW_SSRManifold_batch_size = 1000

SENTENCE_SSRManifold_learning_rate = 0.0001
SENTENCE_SSRManifold_n_epochs = 2500
SENTENCE_SSRManifold_reg_param = 1.0
SENTENCE_SSRManifold_batch_size = 1000

# MULTIRES_WeaSL
MULTIRES_WeaSL_learning_rate = 0.0001
MULTIRES_WeaSL_n_epochs = 2500
MULTIRES_WeaSL_batch_consistency = 1000

# MULTIRES_Augment
MULTIRES_Augment_learning_rate = 0.0001
MULTIRES_Augment_n_epochs = 2500

# MULTIRES_MIL
MULTIRES_MIL_learning_rate = 0.0001
MULTIRES_MIL_n_epochs = 2500
MULTIRES_MIL_reg_param_1 = 1.0
MULTIRES_MIL_reg_param_2 = 1.0
MULTIRES_MIL_reg_param_3 = 1.0
MULTIRES_MIL_batch_consistency = 1000

# MULTIRES_Attention
MULTIRES_Attention_learning_rate = 0.0001
MULTIRES_Attention_n_epochs = 2500
MULTIRES_Attention_reg_param_1 = 1.0
MULTIRES_Attention_reg_param_2 = 1.0
MULTIRES_Attention_reg_param_3 = 0.01
MULTIRES_Attention_batch_consistency = 1000
