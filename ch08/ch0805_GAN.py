# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   deep-learning-with-python-notebooks
@File       :   ch0805_GAN.py
@Version    :   v0.1
@Time       :   2019-11-29 15:05
@License    :   (C)Copyright 2018-2019, zYx.Tom
@Reference  :   《Python 深度学习，Francois Chollet》, Sec0805，P257
@Desc       :   生成式深度学习，生成式对抗网络简介
"""
import os
import sys

import keras
import matplotlib.pyplot as plt
import numpy as np  # pip install numpy<1.17，小于1.17就不会报错
import winsound
from keras import Input, Model
from keras.layers import Conv2D, Conv2DTranspose, Dropout, LeakyReLU, Reshape
from keras.layers import Dense, Flatten
from keras.losses import binary_crossentropy
from keras.optimizers import rmsprop
from keras.preprocessing import image

# 屏蔽警告：Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# 设置数据显示的精确度为小数点后3位
np.set_printoptions(precision = 3, suppress = True, threshold = np.inf, linewidth = 200)
# to make this notebook's output stable across runs
np.random.seed(42)
# Python ≥3.5 is required
assert sys.version_info >= (3, 5)
# numpy 1.16.4 is required
assert np.__version__ in ["1.16.5", "1.16.4"]
# ----------------------------------------------------------------------
print("Listing 8.29 GAN 生成器网络")
latent_dim = 32
height, width, channels = 32, 32, 3

generator_input = Input(shape = (latent_dim,))
# 将输入转换为大小为（16，16）的128个通道的特征图
x = Dense(128 * 16 * 16)(generator_input)
x = LeakyReLU()(x)
x = Reshape((16, 16, 128))(x)

x = Conv2D(256, 5, padding = 'same')(x)
x = LeakyReLU()(x)

# 上采样为（32，32）
x = Conv2DTranspose(256, 4, strides = 2, padding = 'same')(x)
x = LeakyReLU()(x)

x = Conv2D(256, 5, padding = 'same')(x)
x = LeakyReLU()(x)
x = Conv2D(256, 5, padding = 'same')(x)
x = LeakyReLU()(x)

# 生成一个形状为（32，32）的单通道特征图（即CIFAR10图像的形状）
x = Conv2D(channels, 7, activation = 'tanh', padding = 'same')(x)
# 将生成器模型实例化，将形状为（latend_dim,）的输入映射到形状为（32，32，3）的图像
generator = Model(generator_input, x)
generator.summary()
# ----------------------------------------------------------------------
print("Listing 8.30 GAN 判别器网络")
discriminator_input = Input(shape = (height, width, channels))
x = Conv2D(128, 3)(discriminator_input)
x = LeakyReLU()(x)
x = Conv2D(128, 4, strides = 2)(x)
x = LeakyReLU()(x)
x = Conv2D(128, 4, strides = 2)(x)
x = LeakyReLU()(x)
x = Conv2D(128, 4, strides = 2)(x)
x = LeakyReLU()(x)
x = Flatten()(x)

# 使用 dropout 层（重要技巧）
x = Dropout(0.4)(x)
# 分类层
x = Dense(1, activation = 'sigmoid')(x)
# 将判别器模型实例化，将形状为（32，32，3）的输入转换为二进制的分类决策（真/假）
discriminator = Model(discriminator_input, x)
discriminator.summary()

# 在优化器中使用梯度裁剪（clipvalue用于限制梯度值的范围），
# 为了稳定训练过程，使用学习率衰减（decay表示误差的速度）
discriminator_optimizer = rmsprop(lr = 8e-4, clipvalue = 1.0, decay = 1e-8)
discriminator.compile(optimizer = discriminator_optimizer, loss = binary_crossentropy)
# ----------------------------------------------------------------------
print("Listing 8.31 对抗网络")
# 将判别器模型的权重设置为不可训练（即判别器模型仅应用于 对抗模型）
discriminator.trainable = False

gan_input = Input(shape = (latent_dim,))
gan_output = discriminator(generator(gan_input))
gan = Model(gan_input, gan_output)

gan_optimizer = rmsprop(lr = 4e-4, clipvalue = 1.0, decay = 1e-8)
gan.compile(optimizer = gan_optimizer, loss = binary_crossentropy)
# ----------------------------------------------------------------------
print("Listing 8.32 实现 GAN 的训练")
(x_train, y_train), (_, _) = keras.datasets.cifar10.load_data()
x_train = x_train[y_train.flatten() == 6]  # 选择青蛙图像（编号为6）
# 将数据转换成新的形状，再进行标准化
x_train = x_train.reshape((x_train.shape[0],) + (height, width, channels)).astype('float32') / 255.

iterations, batch_size = 10000, 20
start = 0
for step in range(iterations):
    # 在潜在空间中采样随机点
    random_latent_vectors = np.random.normal(size = (batch_size, latent_dim))
    # 将这些点解码为虚假图像
    generated_images = generator.predict(random_latent_vectors)
    stop = start + batch_size
    real_images = x_train[start:stop]
    # 将生成的虚假的图像和原始的真实图像合并到一个列表
    combined_images = np.concatenate([generated_images, real_images])
    # 将生成的虚假的图像的标签和原始的真实图像的标签合并到一个列表
    labels = np.concatenate([np.ones((batch_size, 1)), np.zeros((batch_size, 1))])
    # 向标签中添加随机噪声（重要技巧）
    labels += 0.05 * np.random.random(labels.shape)
    # 训练判别器
    d_loss = discriminator.train_on_batch(combined_images, labels)
    # 在潜在空间中采样随机点
    random_latent_vectors = np.random.normal(size = (batch_size, latent_dim))
    # 合并标签，全部都是“真实图像”（误导的标签）
    misleading_targets = np.zeros((batch_size, 1))
    # 通过 GAN 模型来训练生成器（此时已经冻结了判别器的权重）
    a_loss = gan.train_on_batch(random_latent_vectors, misleading_targets)
    start += batch_size
    if start > len(x_train) - batch_size:
        start = 0
        pass
    # 每隔100步保存并且绘图
    if step % 100 == 0:
        # 保存模型的权重
        gan.save_weights('gan.h5')
        # 输出训练的指标
        print("判别网络的损失:", d_loss)
        print("对抗网络的损失:", a_loss)
        # 保存一张生成的虚假图像
        img = image.array_to_img(generated_images[0] * 255., scale = False)
        img.save('generated_frog' + str(step) + '.png')
        img = image.array_to_img(real_images[0] * 255., scale = False)
        # 保存一张原始的真实图像，用于对比
        img.save('real_frog' + str(step) + '.png')

# ----------------------------------------------------------------------
# 运行结束的提醒
winsound.Beep(600, 500)
if len(plt.get_fignums()) != 0:
    plt.show()
pass
