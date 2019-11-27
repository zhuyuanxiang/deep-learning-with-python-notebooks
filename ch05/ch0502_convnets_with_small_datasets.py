# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   deep-learning-with-python-notebooks
@File       :   ch0502_convnets_with_small_datasets.py
@Version    :   v0.1
@Time       :   2019-11-20 14:31
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

# 屏蔽警告：Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# 设置数据显示的损失为小数点后3位
np.set_printoptions(precision = 3, suppress = True, threshold = np.inf, linewidth = 200)
# to make this notebook's output stable across runs
seed = 42
np.random.seed(seed)
# Python ≥3.5 is required
assert sys.version_info >= (3, 5)
# numpy 1.16.4 is required
assert np.__version__ in ["1.16.5", "1.16.4"]

# dogs-vs-cats 数据集下载地址
# 链接：https://pan.baidu.com/s/13hw4LK8ihR6-6-8mpjLKDA 密码：dmp4
# 从Kaggle下载需要请外国朋友帮助
# 不明白为什么要写个程序整理数据集？
# 拷贝编号为0~999共1000张猫的图片到 train 数据目录中的cats目录下
# 拷贝编号为1000~1499共500张猫的图片到 validation 数据目录中的cats目录下
# 拷贝编号为1500~1999共500张猫的图片到 test 数据目录中的cats目录下
# 对狗的图片同样操作

# 设置数据目录
base_dir = "C:/Users/Administrator/PycharmProjects/Data/small_datasets"
train_dir = os.path.join(base_dir, 'train')
train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')

validation_dir = os.path.join(base_dir, 'validation')
validation_cats_dir = os.path.join(validation_dir, 'cats')
validation_dogs_dir = os.path.join(validation_dir, 'dogs')

# test_dir = os.path.join(base_dir, 'test')
# test_cats_dir = os.path.join(test_dir, 'cats')
# test_dogs_dir = os.path.join(test_dir, 'dogs')

from keras.preprocessing.image import ImageDataGenerator

print("Code 5-7：使用 ImageDataGenerator 从目录中读取图像")
print("\t 将所有图像乘以 1/255 缩放")
train_data_gen = ImageDataGenerator(
        rescale = 1. / 255, rotation_range = 40, width_shift_range = 0.2, height_shift_range = 0.2,
        shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True, fill_mode = 'nearest'
)
# train_data_gen = ImageDataGenerator(rescale = 1. / 255)
validation_data_gen = ImageDataGenerator(rescale = 1. / 255)

print("\t 将所有的图像的大小调整为 150x150，返回了二进制标签")
# generator 是 Python 的生成器技术
# 生成器的输出是 150x150的 RGB 图像（形状为（20，150，150，3）和 二进制标签（形状为（20，））组成的批量。
train_generator = train_data_gen.flow_from_directory(
        train_dir, target_size = (150, 150), batch_size = 20, class_mode = 'binary'
)
validation_generator = validation_data_gen.flow_from_directory(
        validation_dir, target_size = (150, 150), batch_size = 20, class_mode = 'binary'
)
# ImageDataGenerator()函数参数说明
# rotation_range：是角度值（0~180），表示图像随机旋转的角度范围
# width_shift, height_shift：图像在水平或者垂直方向上平移的范围（相对于总宽度或者总高度的比例）
# shear_range：随机错切变换的角度
# zoom_range：随机缩放的范围
# horizontal_flip：随机将一半图像水平翻转（因为现实世界的图像很少水平对称）
# fill_mode：用于填充新创建像素的方法，创建这些新的像素用于填充旋转或者平移产生的像素缺失

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation = activations.relu, input_shape = (150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation = activations.relu))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation = activations.relu))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation = activations.relu))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation = activations.relu))
model.add(layers.Dense(1, activation = activations.sigmoid))

model.compile(loss = losses.binary_crossentropy, optimizer = optimizers.rmsprop(lr = 1e-4),
              metrics = [metrics.binary_accuracy])

# 训练 generator 的数据，使用多处理器并发会发生死锁
history = model.fit_generator(train_generator, steps_per_epoch = 100, epochs = 30, verbose = 2,
                              validation_data = validation_generator, validation_steps = 50)
# 保存训练好的模型
# ToDo:注意每次训练修改这个值，计算时间太长
model.save('cats_and_dogs_small_2.h5')

binary_accuracy = history.history['binary_accuracy']
val_binary_accuracy = history.history['val_binary_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(1, len(loss) + 1)

plt.figure()
plt.plot(epochs_range, binary_accuracy, 'bo', label = "训练集的损失")
plt.plot(epochs_range, val_binary_accuracy, 'r-', label = "验证集的损失")
plt.title("图5-9：训练精度和验证精度")
plt.legend()

plt.figure()
plt.plot(epochs_range, loss, 'bo', label = "训练集的损失")
plt.plot(epochs_range, val_loss, 'r-', label = "验证集的损失")
plt.title("图5-9：训练损失和验证损失")
plt.legend()

# 运行结束的提醒
winsound.Beep(600, 500)
if len(plt.get_fignums()) != 0:
    plt.show()
pass
