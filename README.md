# Multi-view-Regularization-using-Attention-Mechanism

This repository is the implementation for the paper - 'Multi-view-Regularization-using-Attention-Mechanism', which is to appear in the SIAM International Conference on Data Mining (SDM) 2020.

# Abstract
Many real-world phenomena are observed at multiple resolutions. Predictive models designed to predict these phenomena typically consider different resolutions separately. This approach might be limiting in applications where predictions are desired at fine resolutions but available training data is scarce. In this paper, we propose classification algorithms that leverage supervision from coarser resolutions to help train models on finer resolutions. The different resolutions are modeled as different views of the data in a multi-view framework that exploits the complementarity of features across different views to improve models on both views. Unlike traditional multi-view learning problems, the key challenge in our case is that there is no one-to-one correspondence between instances across different views in our case, which requires explicit modeling of the correspondence of instances across resolutions. We propose to use the features of instances at different resolutions to learn the correspondence between instances across resolutions using an attention mechanism.Experiments on the real-world application of mapping urban areas using satellite observations and sentiment classification on text data show the effectiveness of the proposed methods.

# Details of the implementation

1. Data sets - 
Data sets used in the paper can be downloaded here - https://drive.google.com/file/d/15ymp-CpKTyLMH39vGj5oHPngIb62mgm1/view?usp=sharing
Every data set is split into 3 parts, each with its own .numpy file - (train_fine/train_coarse)/test/validation and unlabeled data in the numpy files consistency_fine, consistency_coarse. Each numpy file has a feature variable x (every row is an instance, columns are features) and a label variable y. Coarse and fine resolution training data have separate numpy files

2. Steps to run the Multires code-
All the configurations are saved in RS/SOURCE/config.py. To run using the set configurations follow these steps:
- cd RS/SOURCE/MULTIRES/Attention/
- python (train/validate/test)_model.py

3. Changing base model for Multi Res
While the current implementation uses certain base models, it is trivial to change them to more complex neural networks. For example, this is an example code to build a two layer neural network using the multi-resolution consistency
```
print("BUILD MODEL")
tf.reset_default_graph()
with tf.name_scope('data'):
    n_epoch = tf.placeholder(tf.float32, shape=())
    X_fine = tf.placeholder(tf.float32, [None, num_features_fine], name="fine_res_inputs")
    Y_fine = tf.placeholder(tf.float32, [None, 1], name="fine_res_labels")
    X_coarse = tf.placeholder(tf.float32, [None, num_features_coarse], name="corase_res_inputs")
    Y_coarse = tf.placeholder(tf.float32, [None, 1], name="coarse_res_labels")
    X_fine_consistency= tf.placeholder(tf.float32, [9*config.MULTIRES_Attention_batch_consistency, num_features_fine], name="fine_res_consistency_inputs")
    X_coarse_consistency = tf.placeholder(tf.float32, [config.MULTIRES_Attention_batch_consistency, num_features_coarse], name="coarse_res_consistency_inputs")

with tf.variable_scope("Variables", reuse=tf.AUTO_REUSE):
    W_1_fine = tf.get_variable("Weights_layer_1_fine", [num_features_fine, 6], initializer=tf.contrib.layers.xavier_initializer())
    b_1_fine = tf.get_variable("Biases_layer_1_fine", [6], initializer=tf.zeros_initializer())
    W_2_fine = tf.get_variable("Weights_layer_2_fine", [6, 1], initializer=tf.contrib.layers.xavier_initializer())
    b_2_fine = tf.get_variable("Biases_layer_2_fine", [1], initializer=tf.zeros_initializer())

    W_1_coarse = tf.get_variable("Weights_layer_1_coarse", [num_features_coarse, 6], initializer=tf.contrib.layers.xavier_initializer())
    b_1_coarse = tf.get_variable("Biases_layer_1_coarse", [6], initializer=tf.zeros_initializer())
    W_2_coarse = tf.get_variable("Weights_layer_2_coarse", [6, 1], initializer=tf.contrib.layers.xavier_initializer())
    b_2_coarse = tf.get_variable("Biases_layer_2_coarse", [1], initializer=tf.zeros_initializer())

    W_attention = tf.get_variable("Weights_attention", [12, 1], initializer=tf.contrib.layers.xavier_initializer())

Z_fine = tf.nn.sigmoid(tf.add(tf.matmul(X_fine, W_1_fine, name="multiply_weights"), b_1_fine, name="add_bias"))
Z_fine = tf.nn.sigmoid(tf.add(tf.matmul(Z_fine, W_2_fine, name="multiply_weights"), b_2_fine, name="add_bias"))

Z_coarse = tf.nn.sigmoid(tf.add(tf.matmul(X_coarse, W_1_coarse, name="multiply_weights"), b_1_coarse, name="add_bias"))
Z_coarse = tf.nn.sigmoid(tf.add(tf.matmul(Z_coarse, W_2_coarse, name="multiply_weights"), b_2_coarse, name="add_bias"))

Z_fine_consistency = tf.nn.sigmoid(tf.add(tf.matmul(X_fine_consistency, W_1_fine, name="multiply_weights"), b_1_fine, name="add_bias"))
Z_coarse_consistency = tf.nn.sigmoid(tf.add(tf.matmul(X_coarse_consistency, W_1_coarse, name="multiply_weights"), b_1_coarse, name="add_bias"))
ind = list(itertools.chain.from_iterable(itertools.repeat(x, 9) for x in range(config.MULTIRES_Attention_batch_consistency)))
Z_concat_consistency = tf.concat([tf.gather(Z_coarse_consistency, ind), Z_fine_consistency], axis=-1)
Z_attention_consistency = []
for i in range(config.MULTIRES_Attention_batch_consistency):
    score = tf.matmul(tf.nn.tanh(Z_concat_consistency[i*9:(i+1)*9]), W_attention)
    attention_weights = tf.nn.softmax(score, axis=0)
    context = tf.divide(tf.matmul(tf.transpose(attention_weights), Z_fine_consistency[i*9:(i+1)*9]), tf.reduce_sum(attention_weights))
    Z_attention_consistency.append(context)
Z_attention_consistency = tf.reshape(tf.convert_to_tensor(Z_attention_consistency), (config.MULTIRES_Attention_batch_consistency, Z_fine_consistency.shape[1]))
Z_coarse_consistency = tf.nn.sigmoid(tf.add(tf.matmul(Z_coarse_consistency, W_2_coarse, name="multiply_weights"), b_2_coarse, name="add_bias"))
Z_fine_consistency = tf.nn.sigmoid(tf.add(tf.matmul(Z_attention_consistency, W_2_fine, name="multiply_weights"), b_2_fine, name="add_bias"))

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
```

This section of code must be changed to use different neural network architectures along with multi-resolution consistency. Another example of using LSTM networks with multi-resolution consistency is shown below,
```
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
```

