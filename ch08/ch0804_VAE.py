# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   deep-learning-with-python-notebooks
@File       :   ch0804_VAE.py
@Version    :   v0.1
@Time       :   2019-11-29 10:30
@License    :   (C)Copyright 2018-2019, zYx.Tom
@Reference  :   《Python 深度学习，Francois Chollet》, Sec0804，P249
@Desc       :   生成式深度学习，用变分自编码器生成图像
"""
import os
import sys

import keras
import matplotlib.pyplot as plt
import numpy as np  # pip install numpy<1.17，小于1.17就不会报错
import winsound
from keras.activations import relu, sigmoid
from keras.layers import Conv2D, Conv2DTranspose, Lambda, Reshape
from keras.layers import Dense, Flatten
from keras.losses import binary_crossentropy

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
print("Listing 8.23 VAE 编码器网络")
from keras import Input
from keras import backend as K
from keras.models import Model

img_shape = (28, 28, 1)
batch_size = 16
laten_dim = 2  # 潜在空间的维度：2维平面

input_img = keras.Input(shape = img_shape)
x = Conv2D(32, 3, padding = 'same', activation = 'relu')(input_img)
x = Conv2D(64, 3, padding = 'same', activation = 'relu', strides = (2, 2))(x)
x = Conv2D(64, 3, padding = 'same', activation = 'relu')(x)
x = Conv2D(64, 3, padding = 'same', activation = 'relu')(x)
shape_before_flattening = K.int_shape(x)
x = Flatten()(x)
x = Dense(32, activation = relu)(x)
# 输入的图像最终被编码为这两个参数
z_mean = Dense(laten_dim)(x)
z_log_var = Dense(laten_dim)(x)


# ----------------------------------------------------------------------
def sampling(args):
    print("Listing 8.24 潜在空间采样的函数")
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape = (K.shape(z_mean)[0], laten_dim), mean = 0., stddev = 1.)
    return z_mean + K.exp(z_log_var) * epsilon


z = Lambda(sampling)([z_mean, z_log_var])

# ----------------------------------------------------------------------
print("Listing 8.25 VAE 解码器网络，将潜在空间点映射为图像")
decoder_input = Input(K.int_shape(z)[1:])
# 对输入进行上采样
x = Dense(np.prod(shape_before_flattening[1:]), activation = 'relu')(decoder_input)
# 将 z 转换为特征图，使其形状与编码器模型最后一个 Flattern 层之前的特征图的形状相同
x = Reshape(shape_before_flattening[1:])(x)
# 使用一个 Conv2DTranspose 层 和一个 Conv2D 层，将 z 触须为与原始输入图像具有相同尺寸的特征图
x = Conv2DTranspose(32, 3, padding = 'same', activation = 'relu', strides = (2, 2))(x)
x = Conv2D(1, 3, padding = 'same', activation = sigmoid)(x)
# 将解码器模型实例化，将 decoder_input 转码为解码后的图像
decoder = Model(decoder_input, x)
# 将这个实例应用于 z，以得到解码后的 z
z_decoded = decoder(z)

# ----------------------------------------------------------------------
print("Listing 8.26 用于计算 VAE 损失的自定义层")


class CustomVariationalLayer(keras.layers.Layer):
    def vae_loss(self, x, z_decoded):
        x = K.flatten(x)
        z_decoded = K.flatten(z_decoded)
        xent_loss = binary_crossentropy(x, z_decoded)
        kl_loss = -5e-4 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis = -1)
        return K.mean(xent_loss + kl_loss)

    def call(self, inputs, **kwargs):  # 通过继承这个函数来实现自定义层
        x = inputs[0]
        z_decoded = inputs[1]
        loss = self.vae_loss(x, z_decoded)
        self.add_loss(loss, inputs = inputs)
        return x  # 不需要使用这个输出，但是层必须要有返回值

    pass


y = CustomVariationalLayer()([input_img, z_decoded])

# ----------------------------------------------------------------------
print("Listing 8.27 训练 VAE")
from keras.datasets import mnist

vae = Model(input_img, y)
vae.compile(optimizer = 'rmsprop', loss = None)
vae.summary()

(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_train = x_train.reshape(x_train.shape + (1,))
x_test = x_test.astype('float32') / 255.
x_test = x_test.reshape(x_test.shape + (1,))

vae.fit(x = x_train, y = None, shuffle = True, epochs = 10, batch_size = batch_size,
        validation_data = (x_test, None))
# ----------------------------------------------------------------------
print("Listing 8.28 从二维潜在空间中采样一组点的网格，并且将其解码为图像")
from scipy.stats import norm

n = 15  # 将显示 15 x 15 的数字网格（共255个数字）
digit_size = 28
figure = np.zeros((digit_size * n, digit_size * n))
# 使用 SciPy 的 ppf 函数对线性分隔的坐标进行变换，以生成潜在变量 z 的值（因为潜在空间的先验分布是高斯分布）
grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        z_sample = np.array([[xi, yi]])
        # 将 z 多次重复以构成一个完整的批量
        z_sample = np.tile(z_sample, batch_size).reshape(batch_size, 2)
        # 将批量解码成数字图像
        x_decoded = decoder.predict(z_sample, batch_size = batch_size)
        # 将批量第一个数字的形状从 （28，28，1）转换为（28，28）
        digit = x_decoded[0].reshape(digit_size, digit_size)
        figure[i * digit_size:(i + 1) * digit_size, j * digit_size:(j + 1) * digit_size] = digit
        pass

plt.figure(figsize = (10, 10))
plt.imshow(figure, cmap = 'Greys_r')

# ----------------------------------------------------------------------
# 运行结束的提醒
winsound.Beep(600, 500)
if len(plt.get_fignums()) != 0:
    plt.show()
pass
