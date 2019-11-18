# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   deep-learning-with-python-notebooks
@File       :   ch0202_data_representation.py
@Version    :   v0.1
@Time       :   2019-11-13 10:08
@License    :   (C)Copyright 2018-2019, zYx.Tom
@Reference  :   《Python 深度学习，Francois Chollet》, Sec0202，P23
@Desc       :   神经网络的数学基础，神经网络的数据表示
"""
import os
import sys
import keras
import sklearn
import winsound
import numpy as np  # pip install numpy<1.17，小于1.17就不会报错
import matplotlib.pyplot as plt
from keras.datasets import mnist

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


# 2.2 神经网络的数据表示
# 数据存储在多维的Numpy数组中，也叫张量（tensor）。
# 张量是矩阵向任意维度的推广，张量的维度（dimension）叫做轴（axis）。
# 2.2.5 张量由三个关键属性定义：
#   - 轴的个数（阶）
#   - 形状。整数元组，表示张量沿着每个轴的维度大小（元素个数）
#   - 数据类型。张量中所包含数据的类型。
# 2.2.8 现实世界中的数据张量
#   - 向量数据：2D张量，形状为（samples，features）
#       - 人口统计数据。
#       - 文本文档数据集。
#   - 时间序列数据或者序列数据：3D张量，形状为（samples，timesteps，features）
#       - 股票价格数据集。
#       - 推文数据集。
#   - 图像数据：4D张量，三个维度（高度、宽度、颜色深度）。灰色图像（一个颜色通道），彩色图像（三个颜色通道）
#       - tensorflow 中形状为（samples，height，width，channels）
#       - theano 中形状为（samples，channels，height，width）
#   - 视频数据：5D张量，形状为（samples，frames，height，width，channels）或者（samples，frames，channels，height，width）
#       - 视频是一系列帧组成，第一帧都是一张图像
# # 2.2.9 向量数据
# 2.2.1 Scalars(0D tensors)：仅包含一个数字的张量叫做标量（标量张量、零维张量、0D张量）。
def scalars():
    print('-' * 50)
    x = np.array(12)
    print("x =", x)
    print("x.ndim =", x.ndim)
    pass


# 2.2.2 Vectors(1D tensors)：数字组成的数组叫做向量（一维张量、1D张量），只有一个轴。
def vectors():
    print('-' * 50)
    x = np.array([12])
    print("x =", x)
    print("x.ndim =", x.ndim)
    x = np.array([12, 3, 5, 14])
    print("x =", x)
    print("x.ndim =", x.ndim)
    pass


# 2.2.3 Matrices(2D tensors)：向量组成的数组叫做矩阵（二维张量、2D张量），有两个轴。
def matrices():
    print('-' * 50)
    x = np.array([[12, 3, 5, 14],
                  [1, 15, 4, 13],
                  [7, 90, 23, 5]])
    print("x =", x)
    print("x.ndim =", x.ndim)
    pass


# 2.2.4 higher-dimensional tensors：多个矩阵合成的数组叫3D张量，多个3D张量合成的数组叫做4D张量，以此类推。
def higher_dimensional_tensors():
    print('-' * 50)
    x = np.array([[[5, 78, 2, 34, 0],
                   [6, 79, 3, 35, 1],
                   [7, 80, 4, 36, 2]],
                  [[5, 78, 2, 34, 0],
                   [6, 79, 3, 35, 1],
                   [7, 80, 4, 36, 2]],
                  [[5, 78, 2, 34, 0],
                   [6, 79, 3, 35, 1],
                   [7, 80, 4, 36, 2]]])
    print("x =", x)
    print("x.ndim =", x.ndim)
    pass


# 2.2.6 在Numpy中操作张量
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()


def tensor_slicing():
    my_slice = train_images[10:100]
    print("my_slice.shape =", my_slice.shape)
    my_slice = train_images[10:100, :, :]
    print("my_slice.shape =", my_slice.shape)
    my_slice = train_images[10:100, 0:28, 0:28]
    print("my_slice.shape =", my_slice.shape)
    my_slice = train_images[:, :14, :14]
    print("my_slice.shape =", my_slice.shape)
    my_slice = train_images[:, 7:-7, 7:-7]
    print("my_slice.shape =", my_slice.shape)


# 2.2.7 数据批量：
# 深度学习中所有数据张量的第一个轴（axis=0）都是样本轴，也叫样本维度（samples axis）
# 对于批量张量，第一个轴（axis=0）叫做批量轴（batch axis），也叫做批量维度（batch dimension）。

def data_batch():
    batch = train_images[:128]
    next_batch = train_images[128:256]
    the_n_batch = train_images[128 * n:128 * (n + 1)]


if __name__ == '__main__':
    # scalars()
    # vectors()
    # matrices()
    # higher_dimensional_tensors()

    tensor_slicing()

    # 运行结束的提醒
    winsound.Beep(600, 500)
    if len(plt.get_fignums()) != 0:
        plt.show()
    pass
