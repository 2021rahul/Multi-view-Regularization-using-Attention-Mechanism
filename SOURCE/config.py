#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 10:28:48 2019

@author: ghosh128
"""

import os
#%% FILES INFO
DATASET = "MINNEAPOLIS"
DATA_DIR = os.path.join("../../../DATA", DATASET)
NUMPY_DIR = os.path.join(DATA_DIR, "NUMPY")
RESULT_DIR = os.path.join(DATA_DIR, "RESULT")
MODEL_DIR = os.path.join(DATA_DIR, "MODEL")
#%% DATA INFO
num_strong_landsat = 200
#%% TRAIN INFO
learning_rate = 0.1
n_epochs = 10000

x = 1000.0
lambda1 = 1.0
lambda2 = 1.0
lambda3 = 1.0
batch_consistency = 1000