# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   deep-learning-with-python-notebooks
@File       :   ch0703_advanced_architecture_patterns.py
@Version    :   v0.1
@Time       :   2019-11-27 16:42
@License    :   (C)Copyright 2018-2019, zYx.Tom
@Reference  :   《Python 深度学习，Francois Chollet》, Sec07，P
@Desc       :   高级的深度学习最佳实践，
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np  # pip install numpy<1.17，小于1.17就不会报错
import winsound
from keras.activations import relu, softmax
from keras.layers import BatchNormalization, Conv2D, GlobalAveragePooling2D, MaxPooling2D, SeparableConv2D
from keras.layers import Dense
from keras.losses import categorical_crossentropy
from keras.models import Sequential
from keras.optimizers import rmsprop

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
def batch_normalization():
    conv_model = Sequential()
    conv_model.add(Conv2D(32, 3, activation = relu))
    conv_model.add(BatchNormalization())

    dense_model = Sequential()
    dense_model.add(Dense(32, activation = relu))
    dense_model.add(BatchNormalization())


# ----------------------------------------------------------------------
def depthwise_separable_convolution():
    # 深度可分享卷积（SeparableConv2D）
    height, width, channels = 64, 64, 3
    num_classes = 10

    model = Sequential()
    model.add(SeparableConv2D(32, 3, activation = relu, input_shape = (height, width, channels,)))
    model.add(MaxPooling2D(2))
    model.add(SeparableConv2D(64, 3, activation = relu))
    model.add(SeparableConv2D(128, 3, activation = relu))
    model.add(MaxPooling2D(2))
    model.add(SeparableConv2D(64, 3, activation = relu))
    model.add(SeparableConv2D(128, 3, activation = relu))
    model.add(GlobalAveragePooling2D())

    model.add(Dense(32, activation = relu))
    model.add(Dense(num_classes, activation = softmax))

    model.compile(optimizer = rmsprop(), loss = categorical_crossentropy)


# ----------------------------------------------------------------------
def models_emsemble():
    preds_a = model_a.predict(x_val)
    preds_b = model_b.predict(x_val)
    preds_c = model_c.predict(x_val)
    preds_d = model_d.predict(x_val)

    # 平均值预测
    final_preds = 0.25 * (preds_a + preds_b + preds_c + preds_b)

    # 加权平均预测(0.5,0.25,0.1,0.15) 权重是根据经验学到的
    final_preds = 0.5 * preds_a + 0.25 * preds_b + 0.1 * preds_c + 0.15 * preds_d


# ----------------------------------------------------------------------
# 运行结束的提醒
winsound.Beep(600, 500)
if len(plt.get_fignums()) != 0:
    plt.show()
pass
