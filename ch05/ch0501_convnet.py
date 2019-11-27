# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   deep-learning-with-python-notebooks
@File       :   ch0501_convnet.py
@Version    :   v0.1
@Time       :   2019-11-20 10:18
@License    :   (C)Copyright 2018-2019, zYx.Tom
@Reference  :   《Python 深度学习，Francois Chollet》, Sec05，P
@Desc       :   深度学习用于计算机视觉，卷积神经网络简介
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np  # pip install numpy<1.17，小于1.17就不会报错
import winsound
from keras import activations
from keras import layers
from keras import losses
from keras import metrics
from keras import models
from keras import optimizers
from keras.datasets import mnist
from keras.utils import to_categorical

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


def get_convnet_model():
    print("构造卷积神经网络模型")
    model = models.Sequential()
    # 网络输出张量的形状为：（height, width, channels）
    model.add(layers.Conv2D(32, (3, 3), activation = activations.relu, input_shape = (28, 28, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation = activations.relu))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation = activations.relu))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation = activations.relu))
    model.add(layers.Dense(10, activation = activations.softmax))
    # print(model.summary())
    return model


print("* Code 3-1：加载数据集...")
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print("\t训练数据集（train_labels）：60000 条数据；测试数据集（test_labels）：10000 条数据")
print("\t\t train_images.shape =", train_images.shape)
print("\t\t train_lables.shape =", train_labels.shape)
print("\t\t test_images.shape =", test_images.shape)
print("\t\t test_labels.shape =", test_labels.shape)
print("\t数据集中每条数据是一张图片")
print("\t\t train_images[0].shape =", train_images[0].shape)
print("\t\t test_images[0].shape =", test_images[0].shape)
print("\t每条数据描述一个图片对应的数字：0~9")
print("\t\t train_lables[:10] =", train_labels[:10])
print("\t\t test_labels[:10] =", test_labels[:10])

train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

model = get_convnet_model()
model.compile(optimizer = optimizers.rmsprop(lr = 0.001),
              loss = losses.categorical_crossentropy, metrics = [metrics.categorical_accuracy])
history = model.fit(train_images, train_labels, epochs = 20, batch_size = 64, verbose = 2, use_multiprocessing = True)

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose = 2, use_multiprocessing = True)
print("测试集的评估精度 =", test_acc)

loss = history.history['loss']
epochs_range = range(1, len(loss) + 1)
categorical_acc = history.history['categorical_accuracy']

plt.plot(epochs_range, loss, 'bo', label = "训练集的损失")
plt.title('不同数据集的损失')
plt.xlabel('Epochs--批次')
plt.ylabel('Loss--损失')
plt.legend()

plt.plot(epochs_range, categorical_acc, 'bo', label = "训练集的精确度")
plt.title('不同数据集的精确度')
plt.xlabel('Epochs--批次')
plt.ylabel('Accuracy--精确度')
plt.legend()

# 运行结束的提醒
winsound.Beep(600, 500)
if len(plt.get_fignums()) != 0:
    plt.show()
pass
