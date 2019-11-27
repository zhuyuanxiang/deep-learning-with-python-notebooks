# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   deep-learning-with-python-notebooks
@File       :   ch0504_visualizing_convnet_filters.py
@Version    :   v0.1
@Time       :   2019-11-22 11:54
@License    :   (C)Copyright 2018-2019, zYx.Tom
@Reference  :   《Python 深度学习，Francois Chollet》, Sec05，P
@Desc       :   深度学习用于计算机视觉，卷积神经网络简介
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np  # pip install numpy<1.17，小于1.17就不会报错
import winsound
from keras import backend as K
from keras.applications import VGG16

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

model = VGG16(weights = 'imagenet', include_top = False)


# 将张量转换为图像，方便可视化
def deprocess_image(x):
    # 将张量标准化，均值为0，标准差为0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)  # 1e-5防止除以0
    x *= 0.1

    # 将张量平移，然后剪切超过[0,1]的值
    x += 0.5
    x = np.clip(x, 0, 1)

    # 将张量转换为RGB数组
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x


# Code 5-38：生成过滤器可视化的函数
def generate_pattern(layer_name, filter_index, size = 150):
    # 构建一个损失函数，将该层的第n个过滤器的激活最大化
    layer_output = model.get_layer(layer_name).output
    loss = K.mean(layer_output[:, :, :, filter_index])
    # Code 5-33：获取损失相对于输入的梯度，即计算这个损失相对于输入图像的梯度
    grads = K.gradients(loss, model.input)[0]
    # Code 5-34：梯度标准化
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)  # 加上1e-5，防止除以0
    # Code 5-35：给定 Numpy 输入值，得到 Numpy 输出值，即返回给定输入图像的损失和梯度
    iterate = K.function([model.input], [loss, grads])
    # Code 5-36：通过随机梯度下降让损失最大化，从带有噪声的灰度图像开始
    input_img_data = np.random.random((1, size, size, 3)) * 20 + 128.
    step = 1.
    # 经过40次梯度上升求得最终图像
    for i in range(80):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step
        pass
    img = input_img_data[0]
    return deprocess_image(img)


# plt.figure()
# plt.imshow(generate_pattern('block3_conv1', 0))
# plt.title("图5-29：block3_conv1层第0个通道具有最大响应的模式")
# plt.figure()
# plt.imshow(generate_pattern('block1_conv1', 0, size = 64))
# plt.title("block1_conv1层第0个通道具有最大响应的模式")

def show_layer_filter(layer_name):
    # 每个模式的图像尺寸
    size = 64
    # 两张图片之间的空隙
    margin = 5
    # results是最终要显示的图像，全0代表黑色
    results = np.zeros((8 * size + 7 * margin, 8 * size + 7 * margin, 3))

    for i in range(8):
        for j in range(8):
            horizontal_start = (size + margin) * i
            horizontal_end = horizontal_start + size
            vertical_start = (size + margin) * j
            vertical_end = vertical_start + size
            filter_img = generate_pattern(layer_name, i + (j * 8), size = size)
            results[horizontal_start:horizontal_end, vertical_start:vertical_end, :] = filter_img
            pass
        pass
    plt.figure(figsize = (20, 20))
    plt.imshow(results.astype('uint8'))  # imshow()只能绘制整数（[0,255])和浮点数([0.,1.])
    plt.title("{}层的前64个通道的过滤器模式".format(layer_name))


show_layer_filter('block1_conv1')
show_layer_filter('block2_conv1')
show_layer_filter('block3_conv1')
show_layer_filter('block4_conv1')

# 运行结束的提醒
winsound.Beep(600, 500)
if len(plt.get_fignums()) != 0:
    plt.show()
pass
