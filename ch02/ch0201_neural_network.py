# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   deep-learning-with-python-notebooks
@File       :   ch0201_neural_network.py
@Version    :   v0.1
@Time       :   2019-11-12 18:18
@License    :   (C)Copyright 2018-2019, zYx.Tom
@Reference  :   《Python 深度学习，Francois Chollet》, Sec0201，P21
@Desc       :   神经网络的数学基础，初识神经网络
"""
import os
import sys
import keras
import sklearn
import winsound
import numpy as np  # pip install numpy<1.17，小于1.17就不会报错
import matplotlib.pyplot as plt

# 屏蔽警告：Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# 设置数据显示的精确度为小数点后3位
np.set_printoptions(precision = 3, suppress = True, threshold = np.inf,
                    linewidth = 200)
# to make this notebook's output stable across runs
seed = 42
np.random.seed(seed)
# Python ≥3.5 is required
assert sys.version_info >= (3, 5)
# Scikit-Learn ≥0.20 is required
assert sklearn.__version__ >= "0.20"
# numpy 1.16.4 is required
assert np.__version__ in ["1.16.5", "1.16.4"]

from keras.datasets import mnist

# 加载数据
# 如果数据无法正确下载，可以去网址：https://s3.amazonaws.com/img-datasets/mnist.npz
# 下载后放到C:\Users\Administrator\.keras\datasets目录下。
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print("train_images.shape =", train_images.shape)
print("train_lables.shape =", train_labels.shape)
print("len(train_lables) =", len(train_labels))
print("train_lables[:10] =", train_labels[:10])

print("test_images.shape =", test_images.shape)
print("test_labels.shape =", test_labels.shape)
print("len(train_lables) =", len(train_labels))
print("test_labels[:10] =", test_labels[:10])

from keras import models
from keras import layers

# 网络架构：数据蒸馏（Data Distillation）
network = models.Sequential()
network.add(layers.Dense(512, activation = 'relu', input_shape = (28 * 28,)))
network.add(layers.Dense(10, activation = 'softmax'))

# 编译步骤的三个参数：
#     - 优化器（optimizer）：基于训练数据和损失函数来更新网络的机制
#     - 损失函数（loss function）：网络如何衡量在训练数据上的性能
#     - 指标（metric）：在训练过程和测试过程中需要监控的指标，下面这个例子中使用精度（即正确分类的图像所占的比例）
network.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# 图像数据预处理，将二维图片数据转换为一维数据，再将数据归一化
# train_images = train_images.reshape((60000, 28 * 28)).astype('float32') / 255
# test_images = test_images.reshape((10000, 28 * 28)).astype('float32') / 255
train_images = train_images.reshape((60000, 28 * 28)) / 255.
test_images = test_images.reshape((10000, 28 * 28)) / 255.

# 准备标签（对标签进行分类编码）
from keras.utils import to_categorical

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# 训练网络
network.fit(train_images, train_labels, epochs = 5, batch_size = 128)
test_loss, test_acc = network.evaluate(test_images, test_labels)
print("test_loss:", test_loss)
print('test_acc:', test_acc)

# train_loss, train_acc = network.fit(train_images, train_labels, epochs = 5, batch_size = 128)
# print('train_loss:', train_loss)
# print("train_acc:", train_acc)

# 运行结束的提醒
winsound.Beep(600, 500)
if len(plt.get_fignums()) != 0:
    plt.show()
pass
