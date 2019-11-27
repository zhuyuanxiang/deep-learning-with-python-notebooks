# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   deep-learning-with-python-notebooks
@File       :   ch0603_predict_temperature.py
@Version    :   v0.1
@Time       :   2019-11-24 17:53
@License    :   (C)Copyright 2018-2019, zYx.Tom
@Reference  :   《Python 深度学习，Francois Chollet》, Sec0603，P172
@Desc       :   深度学习用于文本和序列，循环神经网络的高级用法
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np  # pip install numpy<1.17，小于1.17就不会报错
import winsound
from keras.activations import relu
from keras.layers import Bidirectional, Conv1D, GlobalMaxPooling1D, GRU, MaxPooling1D
from keras.layers import Dense, Flatten
from keras.models import Sequential
from keras.optimizers import rmsprop

# 屏蔽警告：Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
from tools import plot_regression_results

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
print("Listing 6.28：加载数据...")
data_dir = "C:/Users/Administrator/PycharmProjects/Data/"
fname = os.path.join(data_dir, 'jena_climate_2009_2016.csv')
f = open(fname, encoding = 'utf-8')
jena_data = f.read()
f.close()

lines = jena_data.split('\n')
header = lines[0].split(',')
class_count = len(header)  # 数据的类别个数
lines = lines[1:]
data_count = len(lines)  # 数据的条数

print("\t数据的表头：", header)
print("\t数据的个数：", data_count)
print("\t每行是一个时间步，记录一个日期和14个与天气有关的值")

# ----------------------------------------------------------------------
print("Listing 6.29：解析数据")
float_data = np.zeros((data_count, class_count - 1))
for i, line in enumerate(lines):
    values = [float(x) for x in line.split(',')[1:]]
    float_data[i, :] = values
    pass
temp = float_data[:, 1]  # 温度（单位摄氏度）


def plot_temperature():
    plt.figure()
    plt.plot(range(len(temp)), temp)
    plt.title("图6-18：在数据集整个时间范围内的温度（单位：摄氏度）")
    pass


def plot_data_point():
    plt.figure()
    plt.plot(range(1440), temp[:1440])
    plt.title("图6-19：数据集中前10天的温度（单位：摄氏度）\n每10分钟记录一个数据，每天144个数据点")
    pass


# ----------------------------------------------------------------------
print("Listing 6.32：数据标准化")
mean = float_data[:200000].mean(axis = 0)
float_data -= mean
std = float_data[:200000].std(axis = 0)
float_data /= std


# ----------------------------------------------------------------------
def generator(data, lookback, delay, min_index, max_index, shuffle = False, batch_size = 128, step = 6):
    """
    :rtype: 元组（samples，targets)
        sampels：输入数据的一个批量
        targets：对应的目标温度数组
    :param data: 浮点数数据组成的原始数组
    :param lookback: 输入数据应该包括过去多少个时间步
    :param delay: 目标应该在未来多少个时间步之后
    :param min_index: data数组中的索引，用于确定需要抽取的时间步的起点
    :param max_index: data数组中的索引，用于确定需要抽取的时间步的终点
    :param shuffle: 是否打乱样本以后再抽取
    :param batch_size: 每个批量的样本数目
    :param step: 数据采样的周期（时间步）。默认为6，表示每个小时抽取一个数据点。
    """
    print("Listing 6.33：生成时间序列样本及其目标的生成器")
    if max_index is None:
        max_index = len(data) - delay - 1
        pass
    global i
    i = min_index + lookback
    while 1:
        if shuffle:
            rows = np.random.randint(min_index + lookback, max_index, size = batch_size)
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback
                pass
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)
            pass
        samples = np.zeros((len(rows), lookback // step, data.shape[-1]))
        targets = np.zeros((len(rows),))
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][1]
            pass
        yield samples, targets
        pass
    pass


print("Listing 6.34：准备训练生成器、验证生成器和测试生成器")
# 每个数据点之间的间隔是10分钟，一天总共要采集144个数据点
step = 6  # 训练数据是按照每小时一个数据点的采样频率从观测数据中提取数据
look_back = 720  # 给定过去5天内的观测数据
delay = 144  # 目标是未来24小时之后的数据
batch_size = 128
epochs = 40
steps_per_epoch = 500
verbose = 2

train_gen = generator(float_data, look_back, delay, min_index = 0, max_index = 200000,
                      shuffle = True, step = step, batch_size = batch_size)
val_gen = generator(float_data, look_back, delay, min_index = 200001, max_index = 300000,
                    shuffle = True, step = step, batch_size = batch_size)
test_gen = generator(float_data, look_back, delay, min_index = 300001, max_index = None,
                     shuffle = True, step = step, batch_size = batch_size)
# val_steps：表示需要从 val_gen 中抽取了多少次，才能保证遍历了完整的验证数据集
val_steps = (30000 - 20001 - look_back) // batch_size
# test_steps:表示需要从 test_gen 中抽取了多少次，才能保证遍历了完整的测试数据集
test_steps = (len(float_data) - 300001 - look_back) // batch_size


# ----------------------------------------------------------------------
def baseline_mae():
    print("Listing 6.35：计算符合常识的基准方法的MAE（预测温度=24小时以前的温度）")
    print("\t温度的平均绝对误差 = mae x temperature_std")
    batch_maes = []
    for _ in range(val_steps):
        samples, targets = next(val_gen)
        preds = samples[:, -1, 1]
        # 平均绝对误差（MAE）指标
        one_point_mae = np.mean(np.abs(preds - targets))
        batch_maes.append(one_point_mae)
        pass
    print("\t基准方法的MAE", np.mean(batch_maes))


# ----------------------------------------------------------------------
def simple_dense():
    print("Listing 6.37：训练并且评估一个密集连接模型")
    model = Sequential(name = "密集连接模型")
    model.add(Flatten(input_shape = (look_back // step, float_data.shape[-1])))
    model.add(Dense(32, activation = relu))
    model.add(Dense(1))
    model.summary()
    model.compile(optimizer = rmsprop(), loss = 'mae')
    history = model.fit_generator(
            train_gen, steps_per_epoch = steps_per_epoch, epochs = epochs, verbose = verbose,
            validation_data = val_gen, validation_steps = val_steps
    )

    print("Listing 6.38：绘制结果")
    title = "图6-20：简单的密集连接网络在温度预测任务"
    plot_regression_results(history, title, epochs)
    pass


# ----------------------------------------------------------------------
def original_gru():
    print("Listing 6.39：训练并且评估一个基于 GRU 的模型")
    model = Sequential(name = "基于 GRU 的模型")
    model.add(GRU(32, input_shape = (None, float_data.shape[-1])))
    model.add(Dense(1))
    model.summary()
    model.compile(optimizer = rmsprop(), loss = 'mae')
    history = model.fit_generator(
            train_gen, steps_per_epoch = steps_per_epoch, epochs = epochs, verbose = verbose,
            validation_data = val_gen, validation_steps = val_steps
    )

    print("绘制基于 GRU 的模型的结果")
    title = "图6-21：使用 GRU 在温度预测任务"
    plot_regression_results(history, title, epochs)
    pass


# ----------------------------------------------------------------------
def gru_dropout():
    print("Listing 6.40：训练并且评估一个使用 Dropout 正则化的基于 GRU 的模型")
    # Theano 不再对 Keras 的 RNN 的 Dropout 提供支持了，需要更换 TensorFlow
    model = Sequential(name = "使用 Dropout 正则化的基于 GRU 的模型")
    model.add(GRU(32, dropout = 0.2, recurrent_dropout = 0.2, input_shape = (None, float_data.shape[-1])))
    model.add(Dense(1))
    model.summary()
    model.compile(optimizer = rmsprop(), loss = 'mae')
    history = model.fit_generator(
            train_gen, steps_per_epoch = steps_per_epoch, epochs = epochs, verbose = verbose,
            validation_data = val_gen, validation_steps = val_steps
    )

    print("绘制使用 Dropout 正则化的基于 GRU 的模型的结果")
    title = "图6-22：使用 Dropout 正则化的基于 GRU 的模型在温度预测任务"
    plot_regression_results(history, title, epochs)
    pass


# ----------------------------------------------------------------------
def gru_dropout_stacking():
    print("Listing 6.41：训练并且评估一个使用 Dropout 正则化的堆叠 GRU 的模型")
    model = Sequential(name = "使用 Dropout 正则化的堆叠 GRU 的模型")
    model.add(GRU(32, dropout = 0.1, recurrent_dropout = 0.5,
                  return_sequences = True, input_shape = (None, float_data.shape[-1])))
    model.add(GRU(64, activation = relu, dropout = 0.1, recurrent_dropout = 0.5))
    model.add(Dense(1))
    model.summary()
    model.compile(optimizer = rmsprop(), loss = 'mae')
    history = model.fit_generator(
            train_gen, steps_per_epoch = steps_per_epoch, epochs = epochs, verbose = verbose,
            validation_data = val_gen, validation_steps = val_steps
    )

    print("绘制使用 Dropout 正则化的堆叠 GRU 的模型的结果")
    title = "图6-22：使用 Dropout 正则化的堆叠 GRU 的模型在温度预测任务"
    plot_regression_results(history, title, epochs)


# ----------------------------------------------------------------------
def bidirectional_gru():
    print("训练并且评估一个使用双向 GRU 的模型")
    model = Sequential(name = "使用双向 GRU 的模型")
    model.add(Bidirectional(GRU(32), input_shape = (None, float_data.shape[-1])))
    model.add(Dense(1))
    model.summary()
    model.compile(optimizer = rmsprop(), loss = 'mae')
    history = model.fit_generator(
            train_gen, steps_per_epoch = steps_per_epoch, epochs = epochs, verbose = verbose,
            validation_data = val_gen, validation_steps = val_steps
    )

    print("绘制使用 Dropout 正则化的基于 GRU 的模型的结果")
    title = "图6-22：使用 Dropout 正则化的基于 GRU 的模型在温度预测任务"
    plot_regression_results(history, title, epochs)
    pass


# ----------------------------------------------------------------------
# 6.4.4 结合 CNN 和 RNN 来处理长序列
def simple_conv1d():
    print("Listing 6.47：在温度预测数据上训练并且评估一个简单的一维卷积神经网络")
    model = Sequential(name = "简单的一维卷积神经网络")
    model.add(Conv1D(32, 5, activation = relu, input_shape = (None, float_data.shape[-1])))
    model.add(MaxPooling1D(3))
    model.add(Conv1D(32, 5, activation = relu))
    model.add(MaxPooling1D(3))
    model.add(Conv1D(32, 5, activation = relu))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(1))
    model.summary()
    model.compile(optimizer = rmsprop(), loss = 'mae')
    history = model.fit_generator(
            train_gen, steps_per_epoch = steps_per_epoch, epochs = epochs, verbose = verbose,
            validation_data = val_gen, validation_steps = val_steps
    )
    title = "图6-29：应用简单的一维卷积神经网络在温度预测数据集"
    plot_regression_results(history, title, epochs)


# ----------------------------------------------------------------------
def conv1d_gru():
    print("Listing 6.48：为温度预测数据集准备更高分辨率的数据生成器")
    # 可以利用 CNN 从更大的数据窗口中进行数据预处理，提取数据的高级特征，再送入 RNN 中计算，不会出现大量增加计算资源的问题
    # 将 原来的6（每小时一个数据点）改成3（每30分钟一个数据点），数据量加倍
    step = 3
    train_gen = generator(float_data, look_back, delay, min_index = 0, max_index = 200000,
                          shuffle = True, step = step, batch_size = batch_size)
    val_gen = generator(float_data, look_back, delay, min_index = 200001, max_index = 300000,
                        shuffle = True, step = step, batch_size = batch_size)
    test_gen = generator(float_data, look_back, delay, min_index = 300001, max_index = None,
                         shuffle = True, step = step, batch_size = batch_size)
    # val_steps：表示需要从 val_gen 中抽取了多少次，才能保证遍历了完整的验证数据集
    val_steps = (30000 - 20001 - look_back) // batch_size
    # test_steps:表示需要从 test_gen 中抽取了多少次，才能保证遍历了完整的测试数据集
    test_steps = (len(float_data) - 300001 - look_back) // batch_size

    print("Listing 6.49：在温度预测数据上训练并且评估一个结合了一维卷积神经网络和 GRU 层的模型")
    model = Sequential(name = "一维卷积神经网络和 GRU 层")
    model.add(Conv1D(32, 5, activation = relu, input_shape = (None, float_data.shape[-1])))
    model.add(MaxPooling1D(3))
    model.add(Conv1D(32, 5, activation = relu))
    model.add(GRU(32, dropout = 0.1, recurrent_dropout = 0.5))
    model.add(Dense(1))
    model.summary()
    model.compile(optimizer = rmsprop(), loss = 'mae')
    history = model.fit_generator(
            train_gen, steps_per_epoch = steps_per_epoch, epochs = epochs, verbose = verbose,
            validation_data = val_gen, validation_steps = val_steps
    )
    title = "图6-31：一维卷积神经网络 + GRU 在温度预测数据集"
    plot_regression_results(history, title, epochs)
    # 从验证损失来看，这种架构的效果不如只用正则化的 GRU 模型，
    # 但是在基于查看了两倍的数据量的条件下，速度要快很多。


# ----------------------------------------------------------------------
# plot_temperature()
# print('*'*50)
# plot_data_point()
# print('*'*50)
# baseline_mae()
# print('*'*50)
# simple_dense()
# print('*'*50)
# original_gru()
# print('*'*50)
gru_dropout()
print('*' * 50)
gru_dropout_stacking()
print('*' * 50)
bidirectional_gru()
# ----------------------------------------------------------------------
# print('*'*50)
# simple_conv1d()
print('*' * 50)
conv1d_gru()
# ----------------------------------------------------------------------
# 运行结束的提醒
winsound.Beep(600, steps_per_epoch)
if len(plt.get_fignums()) != 0:
    plt.show()
pass
