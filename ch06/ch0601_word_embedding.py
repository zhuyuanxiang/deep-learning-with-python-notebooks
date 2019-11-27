# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   deep-learning-with-python-notebooks
@File       :   ch0601_word_embedding.py
@Version    :   v0.1
@Time       :   2019-11-23 12:23
@License    :   (C)Copyright 2018-2019, zYx.Tom
@Reference  :   《Python 深度学习，Francois Chollet》, Sec060102，P151
@Desc       :   深度学习用于文本和序列，处理文本数据——使用词嵌入
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np  # pip install numpy<1.17，小于1.17就不会报错
import winsound
from keras.activations import sigmoid
from keras.datasets import imdb
from keras.layers import Dense, Flatten
from keras.layers import Embedding
from keras.losses import binary_crossentropy
from keras.metrics import binary_accuracy
from keras.models import Sequential
from keras.optimizers import rmsprop
from keras.preprocessing.sequence import pad_sequences

# 屏蔽警告：Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
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

# ToKnown:这里只介绍了 Embedding 的构成方式，与 skip-gram 和 CBOW 并没有明确的关系
# Embedding层的定义至少两个参数（samples，sequence_length）：
# 标记的个数：1000，即最大单词索引+1
# 嵌入的维度：64
# 返回一个形状为（samples，sequence_length，embedding_dimensionality）的三维浮点张量
# 可以使用 RNN 层或者一维卷积层来处理这个三维张量
# 将一个Embedding层实例化时，权重（即标记向量的内部字典）是随机设定的
embedding_layer = Embedding(1000, 64)

# Listing 6.6 加载 IMDB 数据
# max_features：是单词表的大小，
# 总共取9997个不同的单词+1个填充位（0）+1个序列开始（1）+1个未知词（2），
# 不在单词表中的的单词都定义为未知词，具体参考ch0304
max_features = 10000
max_len = 20
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words = max_features)
# pad_sequences() 从尾开始截取max_len个字符，如果太长就拿 0 填充
x_train = pad_sequences(x_train, maxlen = max_len)
# x_train_1 = pad_sequences(x_train, maxlen = 10)
# x_train_2 = pad_sequences(x_train, maxlen = 30)
x_test = pad_sequences(x_test, maxlen = max_len)

# Listing 6.7 在 IMDB 数据集上使用 Embedding 层和分类器
model = Sequential()
model.add(Embedding(10000, 8, input_length = max_len))
model.add(Flatten())
model.add(Dense(1, activation = sigmoid))
model.compile(optimizer = rmsprop(lr = 0.001), loss = binary_crossentropy, metrics = [binary_accuracy])
model.summary()
history = model.fit(x_train, y_train, epochs = 10, batch_size = 32, validation_split = 0.2)

# 运行结束的提醒
winsound.Beep(600, 500)
if len(plt.get_fignums()) != 0:
    plt.show()
pass
