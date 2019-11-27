# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   deep-learning-with-python-notebooks
@File       :   ch0304_two_classes.py
@Version    :   v0.1
@Time       :   2019-11-13 17:17
@License    :   (C)Copyright 2018-2019, zYx.Tom
@Reference  :   《Python 深度学习，Francois Chollet》, Sec0304，P51
@Desc       :   神经网络入门，神经网络解决二分类问题（电影评论分类）
"""
import os
import sys
import keras
import sklearn
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
# Scikit-Learn ≥0.20 is required
assert sklearn.__version__ >= "0.20"
# numpy 1.16.4 is required
assert np.__version__ in ["1.16.5", "1.16.4"]

print("* Code 3-1：加载数据集...")
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words = 10000)
print("\t训练数据集（train_data）：25000条数据；测试数据集（test_data）：25000条数据")
print("\t\tlen(train_data) =", len(train_data))
print("\t\tlen(test_data) =", len(test_data))
print("\t数据集中每条数据是一段评论，字数长度不一样")
print("\t\tlen(train_data[0]) =", len(train_data[0]))
print("\t\tlen(train_data[1]) =", len(train_data[1]))
print("\t\ttrain_data[0] =", train_data[0])
print("\t每条数据对应一个评论结果：正面（positive）为1，负面（negative）为0")
print("\t\ttrain_labels[0] =", train_labels[0])
print("\t\ttrain_labels[1] =", train_labels[1])
print("\t每条评论长度不超过10000个单词，最长评论单词个数 = ", end = '')
print(max([max(sequence) for sequence in train_data]))

print("* 将评论解码为英文单词")
# get_word_index()：将单词映射为整数索引的字典
word_index = imdb.get_word_index()
# 将键值颠倒，实现整数索引映射为单词
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
# 索引减去3，是因为0、1、2分别为“padding”（填充）、“start of sequence”（序列开始）、“unknown”（未知词）保留
decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])


# 将数据进行One-Hot编码，重复数据也只统计一次
def vectorize_sequences(sequences, dimension = 10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
        pass
    return results


# 将训练数据集转换为25000*10000的矩阵，每条数据都有10000个维度，这样所有的数据都规整化为统一大小，方便使用神经网络进行训练
print("* Code 3-2：将整数序列编码为二进制矩阵")
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)
y_train = np.asarray(train_labels, dtype = np.float32)
y_test = np.asarray(test_labels, dtype = np.float32)

del train_data, test_data

# 3.4.3 构建网络
print("* Code 3-3 模式定义")
# 神经元多，越容易捕捉数据的细节，也越容易过拟合；神经元少，收敛速度越快，也就可能丢失细节
# 层数少，收敛速度快，容易陷入局部极值点；层数多，收敛速度慢，容易寻找更多更好的极值点
# sigmoid层前面的神经元个数如果太少，那么判断的细节就不足，也会导致精度无法提升
# 注意：一旦进入过拟合，测试集的精度就无法再提升，只会下降
# 使用字符串定义模式
# model = models.Sequential()
# model.add(layers.Dense(16, activation = 'relu', input_shape = (10000,)))
# model.add(layers.Dense(16, activation = 'relu'))
# model.add(layers.Dense(1, activation = 'sigmoid'))

# 使用类定义模式
# ToSee: 我比较喜欢这种方式，比较符合程序员的风格，可以通过程序进行类名和函数名称的检查
model = models.Sequential()
model.add(layers.Dense(32, activation = activations.relu, input_shape = (10000,)))
model.add(layers.Dense(32, activation = activations.relu))
model.add(layers.Dense(32, activation = activations.relu))
model.add(layers.Dense(16, activation = activations.relu))
model.add(layers.Dense(16, activation = activations.relu))
model.add(layers.Dense(1, activation = activations.sigmoid))

print("* Code 3-4 编译模型")
# 使用字符串配置模型
# model.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])

# 使用类配置模型
# ToSee: 我比较喜欢这种方式，比较符合程序员的风格，可以通过程序进行类名和函数名称的检查
model.compile(optimizer = optimizers.RMSprop(lr = 0.001),
              loss = losses.binary_crossentropy,
              metrics = [metrics.binary_accuracy])

# 3.4.4 验证你的方法
print("* Code 3-7 留出验证集")
x_val = x_train[:10000]
partial_x_train = x_train[10000:]

y_val = y_train[:10000]
partial_y_train = y_train[10000:]

print("* Code 3-8 训练模型")
# epochs=4就够了，随着训练迭代次数增多，训练集的精度越来越高，测试集的精度开始下降，说明训练集过拟合
model.fit(partial_x_train, partial_y_train, epochs = 4, batch_size = 512,
          validation_data = (x_val, y_val), use_multiprocessing = True)
print("\t模型预测")
results = model.evaluate(x_test, y_test)
print("\t损失值 = {}，精确度 = {}".format(results[0], results[1]))
predictions = model.predict(x_test)
print("\t模型预测，前10个预测结果 =")
print(predictions[:10])

history = model.fit(partial_x_train, partial_y_train, epochs = 20, batch_size = 512,
                    validation_data = (x_val, y_val), use_multiprocessing = True)
history_dict = history.history
print("history_dict.keys() =", history_dict.keys())
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs_range = range(1, len(loss_values) + 1)

plt.figure()
plt.plot(epochs_range, loss_values, 'bo', label = '训练集的损失')  # bo 蓝色圆点
plt.plot(epochs_range, val_loss_values, 'b', label = '验证集的损失')  # b 蓝色实线
plt.title('图3-7：训练损失和验证损失')
plt.xlabel('Epochs--批次')
plt.ylabel('Loss--损失')
plt.legend()

acc = history_dict['binary_accuracy']
val_acc = history_dict['val_binary_accuracy']

plt.figure()
plt.plot(epochs_range, acc, 'bo', label = '训练集的精度')
plt.plot(epochs_range, val_acc, 'b', label = '验证集的精度')
plt.title('图3-8：训练精度和验证精度')
plt.xlabel('Epochs--批次')
plt.ylabel('Accuracy--精度')
plt.ylim([0., 1.2])
plt.legend()

# 运行结束的提醒
winsound.Beep(600, 500)
if len(plt.get_fignums()) != 0:
    plt.show()
pass
