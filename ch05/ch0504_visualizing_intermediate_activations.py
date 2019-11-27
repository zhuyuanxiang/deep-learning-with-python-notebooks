# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   deep-learning-with-python-notebooks
@File       :   ch0504_visualizing_intermediate_activations.py
@Version    :   v0.1
@Time       :   2019-11-22 11:11
@License    :   (C)Copyright 2018-2019, zYx.Tom
@Reference  :   《Python 深度学习，Francois Chollet》, Sec05，P
@Desc       :   深度学习用于计算机视觉，卷积神经网络简介
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np  # pip install numpy<1.17，小于1.17就不会报错
import winsound
from keras import models
from keras.models import load_model
from keras.preprocessing import image

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

model = load_model('cats_and_dogs_small_2.h5')
model.summary()

img_path = "C:/Users/Administrator/PycharmProjects/Data/small_datasets/test/cats/cat.1700.jpg"
img = image.load_img(img_path, target_size = (150, 150))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis = 0)
img_tensor /= 255.  # 训练模型的输入数据需要的基本预处理
print("图片的形状：", img_tensor.shape)
plt.imshow(img_tensor[0])
plt.title("图5-24：测试的猫图像")

layer_outputs = [layer.output for layer in model.layers[:8]]  # 提取前8层的输出
# 定义一个模型，给定模型的输入，就可以得到模型的输出
activation_model = models.Model(inputs = model.input, outputs = layer_outputs)
# 使用定义的模型，输入预处理过的图片数据，得到8个Numpy数组组成的列表，表示前8层的输出
activations = activation_model.predict(img_tensor)
first_layer_activation = activations[0]
# 第一个卷积层的激活（经过第一层后的输出）是148x148的特征图，有32个通道
# 不同机器相同模型相同的通道输出的内容不一定相同
print("模型中第一层激活函数（即输出）的形状：", first_layer_activation.shape)
plt.matshow(first_layer_activation[0, :, :, 4], cmap = 'viridis')
plt.title("图5-25：对于测试的猫图像，第一层激活的第4个通道")
plt.matshow(first_layer_activation[0, :, :, 7], cmap = 'viridis')
plt.title("图5-26：对于测试的猫图像，第一层激活的第7个通道")

# 层的名称，用于在绘制图中描述
layer_names = []
for layer in model.layers[:8]:
    layer_names.append(layer.name)
    pass
images_per_row = 16
for layer_name, layer_activation in zip(layer_names, activations):
    # 特征图中的特征个数
    n_features = layer_activation.shape[-1]
    # 特征图的形状（1，size，size，n_features）
    size = layer_activation.shape[1]
    # 在这个矩阵中将激活通道按每行16个图片平铺
    n_cols = n_features // images_per_row
    display_grid = np.zeros((size * n_cols, images_per_row * size))

    # 将每个过滤器作为图片显示在一个大的网格中
    for col in range(n_cols):
        for row in range(images_per_row):
            channel_image = layer_activation[0, :, :, col * images_per_row + row]
            # 对特征进行后处理，使其看起来更加美观
            channel_image -= channel_image.mean()
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[col * size:(col + 1) * size, row * size:(row + 1) * size] = channel_image
            pass
        pass
    scale = 1. / size
    plt.figure(figsize = (scale * display_grid.shape[1], scale * display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect = 'auto', cmap = 'viridis')

# 运行结束的提醒
winsound.Beep(600, 500)
if len(plt.get_fignums()) != 0:
    plt.show()
pass
