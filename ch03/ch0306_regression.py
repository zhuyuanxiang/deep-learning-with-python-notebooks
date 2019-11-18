# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   deep-learning-with-python-notebooks
@File       :   ch0306_regression.py
@Version    :   v0.1
@Time       :   2019-11-18 11:14
@License    :   (C)Copyright 2018-2019, zYx.Tom
@Reference  :   《Python 深度学习，Francois Chollet》, Sec0306，P66
@Desc       :   神经网络入门，神经网络解决
"""
import os
import sys
import sklearn
import winsound
import numpy as np  # pip install numpy<1.17，小于1.17就不会报错
import matplotlib.pyplot as plt
from keras.datasets import boston_housing
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
# Scikit-Learn ≥0.20 is required
assert sklearn.__version__ >= "0.20"
# numpy 1.16.4 is required
assert np.__version__ in ["1.16.5", "1.16.4"]

# 3.6.1 Boston 房价数据集
print("* Code 3-24 加载数据集...")
(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()
print("\t训练数据集（train_data）：404 条数据；测试数据集（test_data）：102 条数据")
print("\t\tlen(train_data) =", len(train_data))
print("\t\tlen(test_data) =", len(test_data))
print("\t数据集中每条数据有13个特征，每个特征值为一个实数")
print("\t\tlen(train_data[0]) =", len(train_data[0]))
print("\t\tlen(train_data[1]) =", len(train_data[1]))
print("\t\ttrain_data[0] =", train_data[0])
print("\t每条标签对应具体的房价")
print("\t\ttrain_targets[0] =", train_targets[0])
print("\t\ttrain_targets[1] =", train_targets[1])

# 3.6.2 准备数据
print("* Code 3-25：数据标准化（零均值，单位方差）")
mean = train_data.mean(axis = 0)
train_data -= mean
std = train_data.std(axis = 0)
train_data /= std

test_data -= mean
test_data /= std


# 3.6.3 构建网络
# Code 3-26：模型定义
# MSE（Mean Squared Error，均方误差）：预测值与目标值之差的平方，回归问题常用的损失函数
# MAE（Mean Absolute Error，平均绝对误差）：预测值与目标值之差的绝对值。
def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation = activations.relu, input_shape = (train_data.shape[1],)))
    model.add(layers.Dense(64, activation = activations.relu))
    model.add(layers.Dense(1))
    model.compile(optimizer = optimizers.rmsprop(lr = 0.001),
                  loss = losses.mse, metrics = [metrics.mae])
    return model


# 3.6.4 K 折验证
print("* Code 3-27：K 折验证（保存每折的验证结果）")
k = 4
num_val_samples = len(train_data) // k
num_epochs = 500
all_mae_histories = []
all_scores = []
for i in range(k):
    print("\tprocessing fold #", i)

    val_start_index = i * num_val_samples
    val_end_index = (i + 1) * num_val_samples
    val_data = train_data[val_start_index: val_end_index]
    val_targets = train_targets[val_start_index: val_end_index]

    partial_train_data = np.concatenate([train_data[:val_start_index], train_data[val_end_index:]])
    partial_train_targets = np.concatenate([train_targets[:val_start_index], train_targets[val_end_index:]])

    model = build_model()
    history = model.fit(partial_train_data, partial_train_targets,
                        validation_data = (val_data, val_targets),
                        epochs = num_epochs, batch_size = 1, verbose = 0, use_multiprocessing = True)
    mae_history = history.history['val_mean_absolute_error']
    all_mae_histories.append(mae_history)
    val_mse, val_mae = model.evaluate(val_data, val_targets, verbose = 0)
    all_scores.append(val_mae)
    pass
print("\t四轮交叉验证对测试集验证的MAE：")
print(all_scores)
print("\t四轮交叉验证对测试集验证的MAE的平均值：{}".format(np.mean(all_scores)))

print("* Code 3-29：计算所有轮次中的 K 折验证分数平均值")
average_mae_history = [np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]
print("* Code 3-30：绘制验证分数")
plt.figure()
plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
plt.xlabel("迭代次数")
plt.ylabel("验证集的平均MAE")


# 将每个数据点替换为前面数据点的指数移动平均值，以得到光滑的曲线
def smooth_curve(points, factor = 0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append((previous * factor + point * (1 - factor)))
        else:
            smoothed_points.append(point)
            pass
        pass
    return smoothed_points


print("* Code 3-31：绘制验证分数（删除前10个数据点）")
smooth_mae_history = smooth_curve(average_mae_history[10:])
plt.figure()
plt.plot(range(1, len(smooth_mae_history) + 1), smooth_mae_history)
plt.xlabel("迭代次数")
plt.ylabel("验证集的平均MAE")

# 运行结束的提醒
winsound.Beep(600, 500)
if len(plt.get_fignums()) != 0:
    plt.show()
pass
