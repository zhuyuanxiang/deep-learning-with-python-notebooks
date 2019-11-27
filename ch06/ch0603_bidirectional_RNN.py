# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   deep-learning-with-python-notebooks
@File       :   ch0603_bidirectional_RNN.py
@Version    :   v0.1
@Time       :   2019-11-25 17:55
@License    :   (C)Copyright 2018-2019, zYx.Tom
@Reference  :   《Python 深度学习，Francois Chollet》, Sec060308，P184
@Desc       :   深度学习用于文本和序列，使用双向RNN
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np  # pip install numpy<1.17，小于1.17就不会报错
import winsound
from keras.activations import sigmoid
from keras.datasets import imdb
from keras.layers import Bidirectional, Embedding, GRU, LSTM
from keras.layers import Dense
from keras.losses import binary_crossentropy
from keras.metrics import binary_accuracy
from keras.models import Sequential
from keras.optimizers import rmsprop
from keras.preprocessing.sequence import pad_sequences

# 屏蔽警告：Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
from tools import plot_classes_results

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# 设置数据显示的精确度为小数点后3位
np.set_printoptions(precision = 3, suppress = True, threshold = np.inf, linewidth = 200)
# to make this notebook's output stable across runs
seed = 42
np.random.seed(seed)
# Python ≥3.5 is required
assert sys.version_info >= (3, 5)
# numpy 1.16.4 is required
assert np.__version__ in ["1.16.5", "1.16.4"]
# ----------------------------------------------------------------------
verbose = 2
validation_split = 0.2
epochs = 20
batch_size = 128
max_features = 10000
max_len = 500
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words = max_features)
# 将训练数据序列反转，反转的是每个评论里面的内容，评论的顺序并没有反转，因此标签也不需要反转
x_train_reversed = [x[::-1] for x in x_train]
x_test_reversed = [x[::-1] for x in x_test]

x_train = pad_sequences(x_train, maxlen = max_len)
x_test = pad_sequences(x_test, maxlen = max_len)
x_train_reversed = pad_sequences(x_train_reversed, maxlen = max_len)
x_test_reversed = pad_sequences(x_test_reversed, maxlen = max_len)


# ----------------------------------------------------------------------
def reversed_lstm():
    print("Listing 6.42：使用逆序序列训练并且评估一个 LSTM 模型")
    title = "使用逆序序列的 LSTM 模型"
    model = Sequential(name = title)
    model.add(Embedding(max_features, 128))
    model.add(LSTM(32))
    model.add(Dense(1, activation = sigmoid))
    model.summary()
    model.compile(optimizer = rmsprop(lr = 0.001), loss = binary_crossentropy, metrics = [binary_accuracy])
    history = model.fit(x_train_reversed, y_train, epochs = epochs, batch_size = batch_size,
                        validation_split = validation_split, verbose = verbose, use_multiprocessing = True)
    plot_classes_results(history, title, epochs)
    pass


# ----------------------------------------------------------------------
def bidirectional_lstm():
    print("Listing 6.43：训练并且评估一个双向 LSTM")
    title = "使用双向 LSTM 模型"
    model = Sequential(name = title)
    model.add(Embedding(max_features, 32))
    model.add(Bidirectional(LSTM(32)))
    model.add(Dense(1, activation = sigmoid))
    model.summary()
    model.compile(optimizer = rmsprop(lr = 0.001), loss = binary_crossentropy, metrics = [binary_accuracy])
    history = model.fit(x_train, y_train, epochs = epochs, batch_size = batch_size,
                        validation_split = validation_split, verbose = verbose, use_multiprocessing = True)
    plot_classes_results(history, title, epochs)
    pass


# ----------------------------------------------------------------------
def bidirectional_LSTM_dropout():
    print("Listing 6.43：训练并且评估一个使用 dropout 的双向 LSTM")
    title = "使用 dropout 的双向 LSTM 模型"
    model = Sequential(name = title)
    model.add(Embedding(max_features, 32))
    model.add(Bidirectional(LSTM(32, dropout = 0.2, recurrent_dropout = 0.2)))
    model.add(Dense(1, activation = sigmoid))
    model.summary()
    model.compile(optimizer = rmsprop(lr = 0.001), loss = binary_crossentropy, metrics = [binary_accuracy])
    history = model.fit(x_train, y_train, epochs = epochs, batch_size = batch_size,
                        validation_split = validation_split, verbose = verbose, use_multiprocessing = True)
    plot_classes_results(history, title, epochs)
    pass


# ----------------------------------------------------------------------
def bidirectional_GRU():
    print("Listing 6.44：训练并且评估一个双向 GRU")
    title = "使用双向 GRU 模型"
    model = Sequential(name = title)
    model.add(Embedding(max_features, 32))
    model.add(Bidirectional(GRU(32)))
    model.add(Dense(1, activation = sigmoid))
    model.summary()
    model.compile(optimizer = rmsprop(lr = 0.001), loss = binary_crossentropy, metrics = [binary_accuracy])
    history = model.fit(x_train, y_train, epochs = epochs, batch_size = batch_size,
                        validation_split = validation_split, verbose = verbose, use_multiprocessing = True)
    plot_classes_results(history, title, epochs)


# ----------------------------------------------------------------------
reversed_lstm()
bidirectional_lstm()
bidirectional_LSTM_dropout()
bidirectional_GRU()

# 运行结束的提醒
winsound.Beep(600, 500)
if len(plt.get_fignums()) != 0:
    plt.show()
pass
