# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   deep-learning-with-python-notebooks
@File       :   ch0503_pretained_convnet_fine_tuning.py
@Version    :   v0.1
@Time       :   2019-11-22 10:03
@License    :   (C)Copyright 2018-2019, zYx.Tom
@Reference  :   《Python 深度学习，Francois Chollet》, Sec050302，P124
@Desc       :   深度学习用于计算机视觉，使用预训练的卷积神经网络——微调模型
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
from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator

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

conv_base = VGG16(weights = 'imagenet', include_top = False, input_shape = (150, 150, 3))
vgg16_out_dims_product = 4 * 4 * 512

base_dir = "C:/Users/Administrator/PycharmProjects/Data/small_datasets"
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
# train_dir = os.path.join(base_dir, 'tmp_train')
# validation_dir = os.path.join(base_dir, 'tmp_val')
test_dir = os.path.join(base_dir, 'test')

epochs = 30
batch_size = 20
steps = 50
steps_per_epoch = 100
print("导入数据")
train_data_gen = ImageDataGenerator(rescale = 1. / 255, rotation_range = 40, width_shift_range = 0.2,
                                    height_shift_range = 0.2,
                                    shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True, fill_mode = 'nearest')
test_data_gen = ImageDataGenerator(rescale = 1. / 255)

train_generator = train_data_gen.flow_from_directory(
        train_dir, target_size = (150, 150), batch_size = batch_size, class_mode = 'binary'
)
validation_generator = test_data_gen.flow_from_directory(
        validation_dir, target_size = (150, 150), batch_size = batch_size, class_mode = 'binary'
)

print("构造特征提取模型")
model = models.Sequential()
conv_base.trainable = False
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation = activations.relu))
model.add(layers.Dense(1, activation = activations.sigmoid))
print("编译特征提取模型")
model.compile(optimizer = optimizers.rmsprop(lr = 2e-5), loss = losses.binary_crossentropy,
              metrics = [metrics.binary_accuracy])
print("训练特征提取模型")
history = model.fit_generator(train_generator, epochs = epochs, steps_per_epoch = steps_per_epoch,
                              validation_data = validation_generator, validation_steps = 50)

# 模型冻结的设置：
# - 模型的冻结会对所有层直接起作用，但是不会控制每层的冻结的设置
# - 模型每个层的冻结设置可以控制每一层，但是不会对模型冻结产生影响
print("构造冻结微调模型")
print("\t模型参数微调被冻结")
conv_base.trainable = False
print("\t\t模型冻结情况：", conv_base.trainable)
print("\t\t可以训练的参数数目：", len(conv_base.trainable_weights))
print("\t\t可以训练的参数", conv_base.trainable_weights)
print("\t设置模型每一层的参数微调的冻结，但是并不会影响到模型冻结情况，因此模型依然所有参数被冻结")
for layer in conv_base.layers:
    if layer.name != 'block5_conv1':
        layer.trainable = False
print("\t\t模型冻结情况：", conv_base.trainable)
print("\t\t可以训练的参数数目：", len(conv_base.trainable_weights))
print("\t\t可以训练的参数", conv_base.trainable_weights)
print("\t模型参数微调允许，就可以观察到每一层参数微调的冻结情况")
conv_base.trainable = True
print("\t\t模型冻结情况：", conv_base.trainable)
print("\t\t可以训练的参数数目：", len(conv_base.trainable_weights))
print("\t\t可以训练的参数", conv_base.trainable_weights)

print("编译冻结微调模型")
model.compile(optimizer = optimizers.rmsprop(lr = 1e-5), loss = losses.binary_crossentropy,
              metrics = [metrics.binary_accuracy])
print("训练冻结微调模型")
history = model.fit_generator(train_generator, epochs = epochs, steps_per_epoch = steps_per_epoch,
                              validation_data = validation_generator, validation_steps = steps)
print("基于测试集评估冻结微调模型")
test_generator = test_data_gen.flow_from_directory(
        test_dir, target_size = (150, 150), batch_size = batch_size, class_mode = 'binary'
)
test_loss, test_acc = model.evaluate_generator(test_generator, steps = steps)
print("测试集的精确度:", test_acc)


def smooth_curve(points, factor = 0.8):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
            pass
        pass
    return smoothed_points


binary_accuracy = history.history['binary_accuracy']
val_binary_accuracy = history.history['val_binary_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(1, epochs + 1)

plt.plot(epochs_range, smooth_curve(binary_accuracy), 'bo', label = "训练集的精确度")
plt.plot(epochs_range, smooth_curve(val_binary_accuracy), 'r-', label = "验证集的精确度")
plt.title("图5-17：带数据增强的特征提取的训练精确度和验证精确度")
plt.legend()

plt.figure()
plt.plot(epochs_range, smooth_curve(loss), 'bo', label = "训练集的损失")
plt.plot(epochs_range, smooth_curve(val_loss), 'r-', label = "验证集的损失")
plt.title("图5-16：简单特征提取的训练损失和验证损失")
plt.legend()

# 运行结束的提醒
winsound.Beep(600, 500)
if len(plt.get_fignums()) != 0:
    plt.show()
pass
