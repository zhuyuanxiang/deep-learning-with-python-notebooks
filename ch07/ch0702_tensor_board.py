# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   deep-learning-with-python-notebooks
@File       :   ch0702_tensor_board.py
@Version    :   v0.1
@Time       :   2019-11-27 15:26
@License    :   (C)Copyright 2018-2019, zYx.Tom
@Reference  :   《Python 深度学习，Francois Chollet》, Sec070202，P212
@Desc       :   高级的深度学习最佳实践，使用TensorBoard来检查并且监控深度学习模型
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
# ----------------------------------------------------------------------
max_features = 2000
max_len = 500
embedding_size = 128
epochs = 15
batch_size = 128
verbose = 2
validation_split = 0.2

print("Listing 7.7：准备 IMDB 数据集...")
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words = max_features)
x_train = x_train[0:max_features]
x_test = x_test[0:max_features]
y_train = y_train[0:max_features]
y_test = y_test[0:max_features]
print('\t', len(x_train), 'train sequences（训练序列）')
print('\t', len(x_test), 'test sequences（测试序列）')
print('Pad sequences (samples x time)')
x_train = pad_sequences(x_train, maxlen = max_len)
x_test = pad_sequences(x_test, maxlen = max_len)
print('\t x_train shape:', x_train.shape)
print('\t x_test shape:', x_test.shape)
# ----------------------------------------------------------------------
model = Sequential()
model.add(Embedding(max_features, 128, input_length = max_len, name = 'Embedding'))
model.add(Conv1D(32, 7, activation = relu))
model.add(MaxPooling1D(5))
model.add(Conv1D(32, 7, activation = relu))
model.add(GlobalMaxPooling1D())
model.add(Dense(1))
model.summary()
model.compile(optimizer = rmsprop(), loss = binary_crossentropy, metrics = ['acc'])

from keras.utils import plot_model

plot_model(model, to_file = 'model.png')
plot_model(model, show_shapes = True, to_file = 'model_with_parameter.png')

# ----------------------------------------------------------------------
# callbacks = [
#         TensorBoard(
#                 log_dir = 'my_log_dir',  # 日志文件保存的位置
#                 histogram_freq = 1,  # 每一轮之后记录激活直方图
#                 # ToDo:还需要提供 embeddings_data 才能记录数据
#                 # embeddings_freq = 1,  # 每一轮之后记录嵌入数据
#         )
# ]
# history = model.fit(x_train, y_train, epochs = 20, batch_size = 128, validation_split = 0.2,
#                     callbacks = callbacks, verbose = 2, use_multiprocessing = True)
# ----------------------------------------------------------------------
# 运行结束的提醒
winsound.Beep(600, 500)
if len(plt.get_fignums()) != 0:
    plt.show()
pass
