# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   deep-learning-with-python-notebooks
@File       :   ch0203_tensor_operations.py
@Version    :   v0.1
@Time       :   2019-11-13 11:08
@License    :   (C)Copyright 2018-2019, zYx.Tom
@Reference  :   《Python 深度学习，Francois Chollet》, Sec0203，P29
@Desc       :   神经网络的数学基础，神经网络中张量运算
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


# 2.3 张量运算（tensor operations）：深度神经网络学到的所有变换都可以简化为数值数据张量上的张量运算
# 下面两个等式是等价的。
# keras.layers.Dense(512, activation = 'relu')
# output_data = tf.nn.relu(np.dot(W,input_data) + b)

# 2.3.1 逐元素运算：运算独立地应用于张量中的每个元素
# relu的实现
# z=np.maxmum(z,0.)
# 等价代码
def naive_relu(x):
    assert len(x.shape) == 2

    x = x.copy()
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i, j] = max(x[i, j], 0)
            pass
        pass
    return x


# 矩阵加法的实现
# 等价代码
# z=x+y
def naive_add(x, y):
    assert len(x.shape) == 2
    assert x.shape == y.shape

    x = x.copy()
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i, j] += y[i, j]
            pass
        pass
    return x


# ToSee: 注意这个概念，因为日常对广播这个概念的理解和这里的使用不太一样，容易误解和混淆
# 2.3.2 广播(broadcast)：大维度张量和小维度张量相加时，较小的张量会被广播，以匹配较大张量的形状
# 广播的步骤：
# 1） 向较小的张量添加轴（叫做广播轴），使其ndim与较大的张量相同
# 2） 将较小的张量沿着新轴重复，使其形状与较大的张量相同
# 广播的原始实现（2D张量+1D张量）
def naive_add_matrix_add_vector(x, y):
    assert len(x.shape) == 2
    assert len(y.shape) == 1
    assert x.shape[1] == y.shape[0]

    x = x.copy()
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i, j] += y[j]
            pass
        pass
    return x


# 广播还可以应用于求最大值中
def broadcast_in_maximum():
    x = np.random.random((4, 3, 3, 10))
    y = np.random.random((3, 10))
    z = np.maximum(x, y)
    print("x=")
    print(x)
    print("y=")
    print(y)
    print("z=")
    print(z)
    pass


# 2.3.3 张量点积：也叫张量积（tensor product）
def tensor_product():
    x = np.random.random((3, 10))
    y = np.random.random((10, 3))
    z = np.dot(x, y)
    print("z=")
    print(z)
    pass


# 向量点积的原始实现
def naive_vector_dot(x, y):
    assert len(x.shape) == 1
    assert len(y.shape) == 1
    assert x.shape[0] == y.shape[0]

    z = 0
    for i in range(x.shape[0]):
        z += x[i] * y[i]
        pass
    return z


def naive_matrix_vector_dot_with_vector_dot(x, y):
    assert len(x.shape) == 2
    assert len(y.shape) == 1
    assert x.shape[1] == y.shape[0]

    z = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        z[i] = naive_vector_dot(x[i, :], y)
        pass
    return z


def naive_matrix_vector_dot(x, y):
    assert len(x.shape) == 2
    assert len(y.shape) == 1
    assert x.shape[1] == y.shape[0]

    z = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            z[i] += x[i, j] * y[j]
        pass
    return z


def naive_matrix_dot(x, y):
    assert len(x.shape) == 2
    assert len(y.shape) == 2
    assert x.shape[1] == y.shape[0]

    z = np.zeros(x.shape[0], y.shape[1])
    for i in range(x.shape[0]):
        for j in range(y.shape[1]):
            row_x = x[i, :]
            column_y = y[:, j]
            z[i, j] += naive_vector_dot(x, y)
        pass
    return z


# 2.3.4 张量变形（tensor reshaping）：改变张量的行和列，得到需要的形状。
def tensor_reshapeing():
    x = np.array([[0., 1.], [2., 3.], [4., 5.]])
    print("x =")
    print(x)
    print("x.shape =", x.shape)
    x = x.reshape((6, 1))
    print("x.reshape((6, 1)) =")
    print(x)
    x = x.reshape((2, 3))
    print("x.reshape((2,3)) =")
    print(x)
    x = np.transpose(x)
    print("np.transpose(x) =")
    print(x)
    pass


# 2.3.5 张量运算的几何解释：许多几何操作都可以表示为张量运算。

# 2.3.6 深度学习的几何解释：深度网络的每一层都通过变换使数据（许多层堆叠在一起）解开一点点，从而实现复杂的解开过程。

if __name__ == '__main__':
    # broadcast_in_maximum()
    # tensor_product()
    tensor_reshapeing()

    # 运行结束的提醒
    winsound.Beep(600, 500)
    if len(plt.get_fignums()) != 0:
        plt.show()
    pass
