# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   deep-learning-with-python-notebooks
@File       :   ch0503_pretained_convnet.py
@Version    :   v0.1
@Time       :   2019-11-21 10:47
@License    :   (C)Copyright 2018-2019, zYx.Tom
@Reference  :   《Python 深度学习，Francois Chollet》, Sec0503，P115
@Desc       :   深度学习用于计算机视觉，使用预训练的卷积神经网络
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

# VGG16()的三个参数
# weights：模型初始化的权重检查点
# include_top：指定模型最后是否包含密集连接分类器（ImageNet的密集连接分类器对应于1000个类别），本例中使用自己的密集连接分类器（对应两个类别：cat和dog）
# input_shape：输入到网络中的图像张量的形状。如果不传入参数，网络可以处理任意形状的输入。
conv_base = VGG16(weights = 'imagenet', include_top = False, input_shape = (150, 150, 3))
vgg16_out_dims_product = 4 * 4 * 512
# 注：(4,4,512)是VGG16网络输出的最后一层的维度，具体参考 summary() 函数
# conv_base.summary()

base_dir = "C:/Users/Administrator/PycharmProjects/Data/small_datasets"
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')
epochs = 30
batch_size = 20
data_gen = ImageDataGenerator(rescale = 1. / 255)


def extract_feature(directory, sample_count):
    features = np.zeros(shape = (sample_count, 4, 4, 512))
    labels = np.zeros(shape = (sample_count,))
    generator = data_gen.flow_from_directory(
            directory, target_size = (150, 150), batch_size = batch_size, class_mode = 'binary'
    )
    print("\t 数据总量 = {}，数据处理中...".format(sample_count))
    step = (sample_count // batch_size) // 10
    for i, (inputs_batch, labels_batch) in enumerate(generator):
        if i % step == 0:
            print("\t 正在处理第{}个数据".format(i * batch_size + 1))
        features_batch = conv_base.predict(inputs_batch)
        features[i * batch_size:(i + 1) * batch_size] = features_batch
        labels[i * batch_size:(i + 1) * batch_size] = labels_batch
        # 生成器在循环中不断生成数据，所以必须在读取完所有图像后终止循环
        if (i + 1) * batch_size >= sample_count:
            break
    return features, labels


print("处理验证数据集的图片")
validation_features, validation_labels = extract_feature(validation_dir, 1000)
validation_features = np.reshape(validation_features, (1000, vgg16_out_dims_product))
print("处理训练数据集的图片")
train_features, train_labels = extract_feature(train_dir, 2000)
train_features = np.reshape(train_features, (2000, vgg16_out_dims_product))
# print("处理临时验证数据集的图片，利用临时数据集加快训练速度，发现代码中的问题")
# validation_features, validation_labels = extract_feature(os.path.join(base_dir, 'tmp_val'), 200)
# validation_features = np.reshape(validation_features, (200, vgg16_out_dims_product))
# print("处理临时训练数据集的图片，利用临时数据集加快训练速度，发现代码中的问题")
# train_features, train_labels = extract_feature(os.path.join(base_dir, 'tmp_train'), 200)
# train_features = np.reshape(train_features, (200, vgg16_out_dims_product))
# print("处理测试数据集的图片")
# test_features, test_labels = extract_feature(test_dir, 1000)
# test_features = np.reshape(test_features, (1000, vgg16_out_dims_product))


print("构造模型")
model = models.Sequential()
model.add(layers.Dense(256, activation = activations.relu, input_dim = vgg16_out_dims_product))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation = activations.sigmoid))
model.compile(optimizer = optimizers.rmsprop(lr = 2e-5), loss = losses.binary_crossentropy,
              metrics = [metrics.binary_accuracy])
print("训练模型")
# 密集网络层使用多处理并发计算不会死锁，可能卷积网络层使用并发操作可能会发生死锁
history = model.fit(train_features, train_labels, epochs = epochs, batch_size = batch_size,
                    validation_data = [validation_features, validation_labels], verbose = 2,
                    use_multiprocessing = True)

binary_accuracy = history.history['binary_accuracy']
val_binary_accuracy = history.history['val_binary_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(1, epochs + 1)

plt.plot(epochs_range, binary_accuracy, 'bo', label = "训练集的精确度")
plt.plot(epochs_range, val_binary_accuracy, 'r-', label = "验证集的精确度")
plt.title("图5-15：简单特征提取的训练精度和验证精度")
plt.legend()

plt.figure()
plt.plot(epochs_range, loss, 'bo', label = "训练集的损失")
plt.plot(epochs_range, val_loss, 'r-', label = "验证集的损失")
plt.title("图5-16：简单特征提取的训练损失和验证损失")
plt.legend()

# 运行结束的提醒
winsound.Beep(600, 500)
if len(plt.get_fignums()) != 0:
    plt.show()
pass
