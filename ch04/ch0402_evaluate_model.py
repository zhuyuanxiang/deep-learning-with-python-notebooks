# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   deep-learning-with-python-notebooks
@File       :   ch0402_evaluate_model.py
@Version    :   v0.1
@Time       :   2019-11-18 17:27
@License    :   (C)Copyright 2018-2019, zYx.Tom
@Reference  :   《Python 深度学习，Francois Chollet》, Sec0402，P76
@Desc       :   机器学习基础，评估机器学习模型
"""
import os
import sys
import winsound
import numpy as np  # pip install numpy<1.17，小于1.17就不会报错
import matplotlib.pyplot as plt
from keras.datasets import imdb
from keras import models
from keras import layers
from keras import activations
from keras import optimizers
from keras import losses
from keras import metrics

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


def get_model():
    return None


def hold_out_validation(train_data, test_data, ratio):
    """

    :param train_data: 所有训练数据
    :param test_data: 测试数据集
    :param ratio: 验证数据集的比例
    """
    num_validation_samples = round(len(train_data) * ratio)
    # 打乱训练数据，保证随机性，防止数据的某种特性在某个区域过于集中
    np.random.shuffle(train_data)

    # 定义训练数据集和验证数据集
    training_data = train_data[num_validation_samples:]
    validation_data = train_data[:num_validation_samples]

    # 使用训练集训练数据，使用验证集评估数据
    model = get_model()
    model.train(training_data)
    validation_score = model.evaluate(validation_data)

    # 使用所有训练数据训练模型，再使用测试集数据评估模型
    model = get_model()
    model.train(train_data)
    test_score = model.evaluate(test_data)


def k_fold_validation(train_data, test_data, k = 4):
    num_validation_samples = len(train_data) // k
    np.random.shuffle(train_data)
    validation_scores = []
    for fold in range(k):
        val_start_index = fold * num_validation_samples
        val_end_index = (fold + 1) * num_validation_samples
        training_data = np.concatenate(train_data[:val_start_index], train_data[val_end_index:])
        validation_data = train_data[val_start_index:val_end_index]

        model = get_model()
        model.train(training_data)
        validation_score = model.evaluate(validation_data)
        validation_scores.append(validation_score)
        pass
    mean_val_score = np.average(validation_scores)

    model = get_model()
    model.train(train_data)
    test_score = model.evaluate(test_data)
    pass


def iterated_k_fold_validation(data, ratio = 0.8, p = 8, k = 4):
    for i in range(p):
        np.random.shuffle(data)
        num_test_samples = round(len(data) * ratio)
        train_data = data[num_test_samples:]
        test_data = data[:num_test_samples]
        k_fold_validation(train_data, test_data, k)
        pass
    pass


# 运行结束的提醒
winsound.Beep(600, 500)
if len(plt.get_fignums()) != 0:
    plt.show()
pass
