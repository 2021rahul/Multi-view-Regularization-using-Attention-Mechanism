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
#%%
print("CONSISTENCY")

files = os.listdir(os.path.join(config.REVIEW_DIR, "unsup"))
files = [file for file in files if file.endswith(".txt")]
index = random.sample(range(len(files)), len(files))

consistency_data_coarse = []
consistency_data_fine = []
count = 0
i=0

while count != config.num_consistency:
    with open(os.path.join(config.REVIEW_DIR, "unsup", files[index[i]]), "r", encoding="utf-8") as f:
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
        if num_words_review<config.max_review_length:
            text = text + " " + " ".join((200-num_words_review)*["UNK"])
        consistency_data_coarse.append(text)
        for i,line in enumerate(clean_lines):
            if len(line.split()) < config.max_sentence_length:
                line = line + " " + " ".join((config.max_sentence_length-len(line.split()))*["UNK"])
                clean_lines[i] = line
        while len(clean_lines) < 9:
            clean_lines.append(" ".join(config.max_sentence_length *["UNK"]))
        consistency_data_fine = consistency_data_fine+clean_lines
        count += 1
        if not count%100:
            print(count)
    i += 1

with open(os.path.join(config.REVIEW_DIR, 'consistency_data_coarse.txt'), 'w') as f:
    for line in consistency_data_coarse:
        f.write("%s\n" % line)

with open(os.path.join(config.REVIEW_DIR, 'consistency_data_fine.txt'), 'w') as f:
    for line in consistency_data_fine:
        f.write("%s\n" % line)
