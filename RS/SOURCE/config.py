#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 10:28:48 2019

@author: ghosh128
"""

import os
# FILES INFO
DATASET = "D1"
DATA_DIR = os.path.join("../../../DATA", DATASET)
LANDSAT_DIR = os.path.join(DATA_DIR, "LANDSAT")
SENTINEL_DIR = os.path.join(DATA_DIR, "SENTINEL")
OSM_DIR = os.path.join(DATA_DIR, "OSM")
NUMPY_DIR = os.path.join(DATA_DIR, "NUMPY")
RESULT_DIR = os.path.join(DATA_DIR, "RESULT")
MODEL_DIR = os.path.join(DATA_DIR, "MODEL")

# DATA INFO
landsat_num_train = 1000
landsat_num_validate = 10
landsat_num_test = 1000

sentinel_num_train = 10
sentinel_num_validate = 10
sentinel_num_test = 9000

num_consistency = 100

# OnlyStrong
LANDSAT_OnlyStrong_learning_rate = 0.01
LANDSAT_OnlyStrong_n_epochs = 10000

SENTINEL_OnlyStrong_learning_rate = 0.01
SENTINEL_OnlyStrong_n_epochs = 10000

# SSRManifold
LANDSAT_SSRManifold_learning_rate = 0.01
LANDSAT_SSRManifold_n_epochs = 10000
LANDSAT_SSRManifold_reg_param = 1.0
LANDSAT_SSRManifold_batch_size = 1000

SENTINEL_SSRManifold_learning_rate = 0.01
SENTINEL_SSRManifold_n_epochs = 10000
SENTINEL_SSRManifold_reg_param = 1.0
SENTINEL_SSRManifold_batch_size = 1000

# MULTIRES_WeaSL
MULTIRES_WeaSL_learning_rate = 0.01
MULTIRES_WeaSL_n_epochs = 10000
MULTIRES_WeaSL_batch_consistency = 1000

# MULTIRES_Augment
MULTIRES_Augment_learning_rate = 0.01
MULTIRES_Augment_n_epochs = 10000

# MULTIRES_MIL
MULTIRES_MIL_x = 1000.0
MULTIRES_MIL_learning_rate = 0.01
MULTIRES_MIL_n_epochs = 10000
MULTIRES_MIL_reg_param_1 = 1.0
MULTIRES_MIL_reg_param_2 = 1.0
MULTIRES_MIL_reg_param_3 = 0.001
MULTIRES_MIL_batch_consistency = 1000

# MULTIRES_Attention
MULTIRES_Attention_learning_rate = 0.001
MULTIRES_Attention_n_epochs = 10000
MULTIRES_Attention_reg_param_1 = 1.0
MULTIRES_Attention_reg_param_2 = 1.0
MULTIRES_Attention_reg_param_3 = 5.0
MULTIRES_Attention_batch_consistency = 1000
