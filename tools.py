# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   deep-learning-with-python-notebooks
@File       :   tools.py
@Version    :   v0.1
@Time       :   2019-11-24 16:43
@License    :   (C)Copyright 2018-2019, zYx.Tom
@Reference  :   《Python 深度学习，Francois Chollet》, Sec06，P
@Desc       :   深度学习用于文本和序列，
"""
import os
import sys

import numpy as np  # pip install numpy<1.17，小于1.17就不会报错
import winsound
# 屏蔽警告：Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
from matplotlib import pyplot as plt

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

# 运行结束的提醒
winsound.Beep(600, 500)
if len(plt.get_fignums()) != 0:
    plt.show()
pass


def plot_classes_results(history, title, epochs):
    # 绘制分类问题的结果
    epochs_range = range(1, epochs + 1)

    loss = history.history['loss']
    val_loss = history.history['val_loss']
    plt.figure()
    plt.plot(epochs_range, loss, 'bo', label = "训练集的损失")
    plt.plot(epochs_range, val_loss, 'r-', label = "验证集的损失")
    plt.title("{}时的训练损失和验证损失".format(title))
    plt.legend()

    binary_accuracy = history.history['binary_accuracy']
    val_binary_accuracy = history.history['val_binary_accuracy']
    plt.figure()
    plt.plot(epochs_range, binary_accuracy, 'bo', label = "训练集的精确度")
    plt.plot(epochs_range, val_binary_accuracy, 'r-', label = "验证集的精确度")
    plt.title("{}时的训练精确度和验证精确度".format(title))
    plt.ylim((-0.2, 1.2))
    plt.legend()
    pass


def plot_regression_results(history, title, epochs):
    # 绘制回归问题的结果
    epochs_range = range(1, epochs + 1)

    loss = history.history['loss']
    val_loss = history.history['val_loss']
    plt.figure()
    plt.plot(epochs_range, loss, 'bo', label = "训练集的损失")
    plt.plot(epochs_range, val_loss, 'r-', label = "验证集的损失")
    plt.title("{}时的训练损失和验证损失".format(title))
    plt.legend()
    pass
