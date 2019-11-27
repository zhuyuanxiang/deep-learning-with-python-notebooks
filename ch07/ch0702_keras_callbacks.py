# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   deep-learning-with-python-notebooks
@File       :   ch0702_keras_callbacks.py
@Version    :   v0.1
@Time       :   2019-11-27 11:54
@License    :   (C)Copyright 2018-2019, zYx.Tom
@Reference  :   《Python 深度学习，Francois Chollet》, Sec0702，P210
@Desc       :   高级的深度学习最佳实践，使用Keras回调函数
"""
import os
import sys

import keras
import matplotlib.pyplot as plt
import numpy as np  # pip install numpy<1.17，小于1.17就不会报错
import winsound
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
from keras.losses import binary_crossentropy
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
# 设置模型检查点，监控模型达到要求时就停止训练
def model_check_and_stop():
    callbacks_list = [
            # 监控模型的验证精度，如果精度在多于一轮时间（即两轮）内不再改善，就中断训练
            EarlyStopping(monitor = 'acc', patience = 1),
            # 每轮过后都保存当前权重于目标文件中，如果‘val_loss’没有改善，就不需要覆盖模型文件，目标是始终保存在训练过程中见到的最佳模型
            ModelCheckpoint(filepath = 'my_model.h5', monitor = 'val_loss', save_best_only = True)
    ]
    model = Sequential()
    model.compile(optimizer = rmsprop(), loss = binary_crossentropy, metrics = ['acc'])
    # 由于回调函数要监控“验证集损失”和“验证集精度”，所以需要传入validation_data（验证数据）
    model.fit(x, y, epochs = 10, batch_size = 32, callbacks = callbacks_list, validation_data = (x_val, y_val))


# ----------------------------------------------------------------------
# ReduceLROnPlateau（Reduce learning rate when a metric has stopped improving.）
# 当监控值到达一个稳定水平（Plateau），那么减少学习率
def reduce_lr_on_plateau():
    # 监控模型的“验证集损失”，如果验证损失在10轮内都没有改善，那么就触发这个回调函数，将学习率乘以0.1
    callbacks_list = [ReduceLROnPlateau(monitor = 'val_loss', factor = 0.1, patience = 10)]
    # 由于回调函数要监控“验证集损失”，所以需要传入validation_data（验证数据）
    model.fit(x, y, epochs = 10, batch_size = 32, callbacks = callbacks_list, validation_data = (x_val, y_val))


# ----------------------------------------------------------------------
# 编写自己的回调函数（创建keras.callbacks.Callback子类）
# 类的时间点：
# on_epoch_begin：在每轮开始时被调用
# on_epoch_end：在每轮结束时被调用
# on_batch_begin：在处理每个批量之前被调用
# on_batch_end：在处理每个批量之后被调用
# on_train_begin：在训练开始时被调用
# on_train_end：在训练结束时被调用
# 回调函数可以访问的属性：
# self.model：调用回调函数的模型实例
# self.validation_data：传入fit()作为验证数据的值
class MyCallBack(keras.callbacks.Callback):
    # 在每轮结束后将模型每层的激活保存到硬盘中（格式为 Numpy 数组）
    # 这个激活是对验证集的第一个样本计算得到的
    def set_model(self, model):
        self.model = model
        layer_outputs = [layer.output for layer in model.layers]
        self.activations_model = keras.models.Model(model.input, layer_outputs)
        pass

    def on_epoch_end(self, epoch, logs = None):
        if self.validation_data is None:
            raise RuntimeError("Requires validation_data.")
        # 获取验证数据的第一个输入样本
        validation_sample = self.validation_data[0][0:1]
        activations = self.activations_model.predict(validation_sample)
        f = open('activations_at_epoch_' + str(epoch) + '.npz', 'w')
        np.savez(f, activations)
        f.close()


# ----------------------------------------------------------------------
# 运行结束的提醒
winsound.Beep(600, 500)
if len(plt.get_fignums()) != 0:
    plt.show()
pass
