# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   deep-learning-with-python-notebooks
@File       :   ch0602_recurrent_neural_network.py
@Version    :   v0.1
@Time       :   2019-11-24 16:00
@License    :   (C)Copyright 2018-2019, zYx.Tom
@Reference  :   《Python 深度学习，Francois Chollet》, Sec0602，P162
@Desc       :   深度学习用于文本和序列，理解循环神经网络（并不适用于情感分析，建议看0603进一步理解RNN）
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np  # pip install numpy<1.17，小于1.17就不会报错
import winsound
from keras.activations import relu, sigmoid
from keras.datasets import imdb
from keras.layers import Dense
from keras.layers import Embedding, LSTM, SimpleRNN
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
# Listing 6-21：简单 RNN 的 Numpy 实现
def simple_rnn_use_numpy():
    timesteps = 100  # 输入序列的时间步数
    input_features = 32  # 输入特征空间的维度
    output_features = 64  # 输出特征空间的维度
    # 输入数据：随机噪声，仅仅作为示例
    inputs = np.random.random((timesteps, input_features))
    state_t = np.zeros((output_features,))  # 初始状态：全零向量

    # 创建随机的权重矩阵
    W = np.random.random((output_features, input_features)) / 10
    U = np.random.random((output_features, output_features)) / 10
    b = np.random.random((output_features,)) / 10

    successive_outputs = []
    for input_t in inputs:
        # 当前输出 = 当前输入 + 前一个输出
        output_t = np.tanh(np.dot(W, input_t) + np.dot(U, state_t), +b)
        successive_outputs.append(output_t)  # 将输出保存到一个列表中
        # 更新网络的状态，用于下一个时间步
        state_t = output_t
        pass
    # 最终的输出是一个形状为（timesteps，output_features）的二维张量
    # np.stack() 把数组组成的列表转换成一个二维数组
    final_output_sequence = np.stack(successive_outputs, axis = 0)
    return final_output_sequence


# ----------------------------------------------------------------------
# 简单 RNN 的 Keras 实现
def keras_simplernn():
    model = Sequential(name = "完整的状态序列")
    model.add(Embedding(10000, 32))
    model.add(SimpleRNN(32, return_sequences = True))
    model.summary()

    model = Sequential(name = "最后一个时间步的输出")
    model.add(Embedding(10000, 32))
    model.add(SimpleRNN(32))
    model.summary()

    model = Sequential(name = "多个循环层的逐个堆叠")
    model.add(Embedding(10000, 32))
    model.add(SimpleRNN(32, return_sequences = True))
    model.add(SimpleRNN(32, return_sequences = True))
    model.add(SimpleRNN(32, return_sequences = True))
    model.add(SimpleRNN(32))
    model.summary()
    pass


# 使用 RNN 和 LSTM 模型应用于 IMDB 电影评论分类问题
max_features = 10000
max_len = 500
batch_size = 128
epochs = 10
# 数据集的详细说明参考 ch0304
print("Listing 6.22：加载数据集...")
(train_data, y_train), (test_data, y_test) = imdb.load_data(num_words = max_features)
x_train = pad_sequences(train_data, maxlen = max_len)
x_test = pad_sequences(test_data, maxlen = max_len)


def train_model(model, data, labels):
    return model.fit(data, labels, epochs = epochs, batch_size = batch_size,
                     validation_split = 0.2, verbose = 2, use_multiprocessing = True)


# ----------------------------------------------------------------------
def definite_rnn():
    title = "将 SimpleRNN 应用于 IMDB "
    model = Sequential(name = title)
    model.add(Embedding(max_features, 64))
    model.add(SimpleRNN(64))
    model.add(Dense(1, activation = sigmoid))
    model.summary()
    model.compile(optimizer = rmsprop(lr = 0.001), loss = binary_crossentropy, metrics = [binary_accuracy])
    history = train_model(model, x_train, y_train)
    plot_classes_results(history, title, epochs)
    print(title + "评估测试集", model.evaluate(x_test, y_test, verbose = 2, use_multiprocessing = True))
    pass


# ----------------------------------------------------------------------
def definite_lstm():
    title = "将 LSTM 应用于 IMDB"
    model = Sequential(name = title)
    model.add(Embedding(max_features, 64))
    model.add(LSTM(64))
    model.add(Dense(1, activation = sigmoid))
    model.summary()
    model.compile(optimizer = rmsprop(lr = 0.001), loss = binary_crossentropy, metrics = [binary_accuracy])
    model = definite_rnn(title)
    history = train_model(model, x_train, y_train)
    plot_classes_results(history, title, epochs)
    print(title + "评估测试集", model.evaluate(x_test, y_test, verbose = 2, use_multiprocessing = True))
    pass


# ----------------------------------------------------------------------
# 重构 ch0304 的二分类问题
def vectorize_sequences(sequences, dimension = 10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
        pass
    return results


vector_train_data = vectorize_sequences(train_data, max_features)
vector_test_data = vectorize_sequences(test_data, max_features)
vector_train_labels = np.asarray(y_train)
vector_test_labels = np.asarray(y_test)


# 将数据进行 One-Hot 编码后，准确率比 RNN 和 LSTM 的质量还好(ch0304确认了密集层的效果确实很好）
def definite_dense_for_one_hot():
    title = "将 Dense+One-Hot 应用于 IMDB"
    model = Sequential(name = title)
    model.add(Dense(16, activation = relu, input_shape = (10000,)))
    model.add(Dense(16, activation = relu))
    model.add(Dense(1, activation = sigmoid))
    model.summary()
    model.compile(optimizer = rmsprop(lr = 0.001), loss = binary_crossentropy, metrics = [binary_accuracy])
    history = train_model(model, vector_train_data, vector_train_labels)
    plot_classes_results(history, title, epochs)
    print(title + "评估测试集",
          model.evaluate(vector_test_data, vector_test_labels, verbose = 2, use_multiprocessing = True))
    pass


# 没有将数据进行 One-Hot 编码，准确率下降的会很厉害
def definite_dense():
    title = "将 Dense 应用于 IMDB"
    model = Sequential(name = title)
    model.add(Dense(16, activation = relu, input_shape = (500,)))
    model.add(Dense(16, activation = relu))
    model.add(Dense(1, activation = sigmoid))
    model.summary()
    model.compile(optimizer = rmsprop(lr = 0.001), loss = binary_crossentropy, metrics = [binary_accuracy])
    history = train_model(model, x_train, y_train)
    plot_classes_results(history, title, epochs)
    print(title + "评估测试集", model.evaluate(x_test, y_test, verbose = 2, use_multiprocessing = True))
    pass


# ----------------------------------------------------------------------
definite_rnn()
definite_lstm()
definite_dense_for_one_hot()
definite_dense()
# 运行结束的提醒
winsound.Beep(600, 500)
if len(plt.get_fignums()) != 0:
    plt.show()
pass
