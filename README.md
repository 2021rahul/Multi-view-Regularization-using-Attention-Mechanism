# Multi-view-Regularization-using-Attention-Mechanism

This repository is the implementation for the paper - 'Multi-view-Regularization-using-Attention-Mechanism', which is to appear in the SIAM International Conference on Data Mining (SDM) 2020.

# Abstract
Many real-world phenomena are observed at multiple resolutions. Predictive models designed to predict these phenomena typically consider different resolutions separately. This approach might be limiting in applications where predictions are desired at fine resolutions but available training data is scarce. In this paper, we propose classification algorithms that leverage supervision from coarser resolutions to help train models on finer resolutions. The different resolutions are modeled as different views of the data in a multi-view framework that exploits the complementarity of features across different views to improve models on both views. Unlike traditional multi-view learning problems, the key challenge in our case is that there is no one-to-one correspondence between instances across different views in our case, which requires explicit modeling of the correspondence of instances across resolutions. We propose to use the features of instances at different resolutions to learn the correspondence between instances across resolutions using an attention mechanism.Experiments on the real-world application of mapping urban areas using satellite observations and sentiment classification on text data show the effectiveness of the proposed methods.

# Details of the implementation

1. Data sets - 
Data sets used in the paper can be downloaded here - https://drive.google.com/file/d/15ymp-CpKTyLMH39vGj5oHPngIb62mgm1/view?usp=sharing
Every data set is split into 3 parts, each with its own .numpy file - (train_fine/train_coarse)/test/validation and unlabeled data in the numpy files consistency_fine, consistency_coarse
Each numpy file has a feature variable x (every row is an instance, columns are features) and a label variable y
Coarse and fine resolution training data have separate numpy files

2. learning MultiRes on a data set


3. Changing base model for Multi Res
While the current implementation uses certain base models, it is trivial to change them to more complex neural networks. Here is how you do it ...

