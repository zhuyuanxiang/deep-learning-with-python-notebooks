# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   deep-learning-with-python-notebooks
@File       :   ch0604_conv1D.py
@Version    :   v0.1
@Time       :   2019-11-26 11:08
@License    :   (C)Copyright 2018-2019, zYx.Tom
@Reference  :   《Python 深度学习，Francois Chollet》, Sec0604，P188
@Desc       :   深度学习用于文本和序列，用卷积神经网络处理序列
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np  # pip install numpy<1.17，小于1.17就不会报错
import winsound
from keras.activations import relu
from keras.datasets import imdb
from keras.layers import Conv1D, Embedding, GlobalMaxPooling1D, MaxPooling1D
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
max_features = 10000
max_len = 500
embedding_size = 128
epochs = 15
batch_size = 128
verbose = 2
validation_split = 0.2

print("Listing 6.45：准备 IMDB 数据集...")
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words = max_features)
print('\t', len(x_train), 'train sequences（训练序列）')
print('\t', len(x_test), 'test sequences（测试序列）')
print('Pad sequences (samples x time)')
x_train = pad_sequences(x_train, maxlen = max_len)
x_test = pad_sequences(x_test, maxlen = max_len)
print('\t x_train shape:', x_train.shape)
print('\t x_test shape:', x_test.shape)


# ----------------------------------------------------------------------
def simple_conv1d():
    print("Listing 6.46：在 IMDB 数据上训练并且评估一个简单的一维卷积神经网络")
    model = Sequential(name = "简单的一维卷积神经网络")
    model.add(Embedding(max_features, embedding_size, input_length = max_len))
    model.add(Conv1D(32, 7, activation = relu))
    model.add(MaxPooling1D(5))
    model.add(Conv1D(32, 7, activation = relu))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(1))
    model.summary()
    model.compile(optimizer = rmsprop(lr = 1e-4), loss = binary_crossentropy, metrics = [binary_accuracy])
    history = model.fit(x_train, y_train, epochs = epochs, batch_size = batch_size,
                        validation_split = validation_split, verbose = verbose, use_multiprocessing = True)
    title = "应用简单的一维卷积神经网络在 IMDB 数据集"
    plot_classes_results(history, title, epochs)
    pass


# ----------------------------------------------------------------------
simple_conv1d()

# 6.4.4 结合 CNN 和 RNN 来处理长序列
# 因为使用的是温度数据集，因此实现在 ch0603_predict_temperature.py 中，方便对比
# 运行结束的提醒
winsound.Beep(600, 500)
if len(plt.get_fignums()) != 0:
    plt.show()
pass
