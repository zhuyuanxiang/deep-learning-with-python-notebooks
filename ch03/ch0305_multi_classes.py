# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   deep-learning-with-python-notebooks
@File       :   ch0305_multi_classes.py
@Version    :   v0.1
@Time       :   2019-11-15 11:45
@License    :   (C)Copyright 2018-2019, zYx.Tom
@Reference  :   《Python 深度学习，Francois Chollet》, Sec0305，P59
@Desc       :   神经网络入门，神经网络解决多分类问题（新闻分类——单标签、多分类问题）
"""
import os
import sys
import sklearn
import winsound
import numpy as np  # pip install numpy<1.17，小于1.17就不会报错
import matplotlib.pyplot as plt
import keras
# from tensorflow import keras      # keras 也可以使用高版本的TensorFlow自带的
from keras.datasets import reuters

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

print("* Code 3-12 加载数据集...")
(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words = 10000)
print("\t训练数据集（train_data）：8982 条数据；测试数据集（test_data）：2246 条数据")
print("\t\tlen(train_data) =", len(train_data))
print("\t\tlen(test_data) =", len(test_data))
print("\t数据集中每条数据是一篇文章，单词个数不一样")
print("\t\tlen(train_data[0]) =", len(train_data[0]))
print("\t\tlen(train_data[1]) =", len(train_data[1]))
print("\t\ttrain_data[0] =", train_data[0])
print("\t每条标签对应一个类别")
print("\t\ttrain_labels[0] =", train_labels[0])
print("\t\ttrain_labels[1] =", train_labels[1])
print("\t每篇文章长度不超过10000个单词，最长文章单词个数 = ", end = '')
print(max([max(sequence) for sequence in train_data]))

print("* Code 3-13：将索引解码为新闻文本")
word_index = reuters.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
# 索引减去3，是因为0、1、2分别为“padding”（填充）、“start of sequence”（序列开始）、“unknown”（未知词）保留
decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])


def vectorize_sequences(sequences, dimension = 10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
        pass
    return results


# 将训练数据集转换为 8982 * 10000 的矩阵，每条数据都有 10000 个维度，这样所有的数据都规整化为统一大小，方便使用神经网络进行训练
print("* Code 3-14：将整数序列编码为二进制矩阵")
print("\t将数据进行One-Hot编码，重复数据也只统计一次")
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

# 标签向量化方法：
# 1. 标签列表转换为整数张量
# 2. 标签列表转换为 One-Hot 编码
print("\t基于 Keras 的 One-Hot 编码方法实现标签向量化")
one_hot_train_labels = keras.utils.to_categorical(train_labels)
one_hot_test_labels = keras.utils.to_categorical(test_labels)

# print("One-Hot 编码的原始实现")
# def to_one_hot(labels, dimension = 46):
#     results = np.zeros((len(labels), dimension))
#     for i, labels in enumerate(labels):
#         results[i, labels] = 1.
#         pass
#     return results
#
#
# one_hot_train_labels = to_one_hot(train_labels)
# one_hot_test_labels = to_one_hot(test_labels)

# 3.5.3 构建网络
# 如果中间层维度过小，会造成信息压缩而导致损失
print("* Code 3-15 模式定义--使用类定义模式")
# ToSee: 我比较喜欢这种方式，比较符合程序员的风格，可以通过程序进行类名和函数名称的检查
model = keras.Sequential()
model.add(keras.layers.Dense(64, activation = keras.activations.relu, input_shape = (10000,)))
model.add(keras.layers.Dense(64, activation = keras.activations.relu))
model.add(keras.layers.Dense(46, activation = keras.activations.softmax))

print("* Code 3-16 编译模型")
# 使用类配置模型
model.compile(optimizer = keras.optimizers.RMSprop(lr = 0.001),
              loss = keras.losses.categorical_crossentropy,
              metrics = [keras.metrics.binary_accuracy])

# 3.5.4 验证你的方法
print("* Code 3-17 留出验证集")
split_number = 8000  # 切分训练集和验证集，修改切分值是因为我的训练精度太高了（99%）
x_val = x_train[:split_number]
partial_x_train = x_train[split_number:]

y_val = one_hot_train_labels[:split_number]
partial_y_train = one_hot_train_labels[split_number:]

print("\tlen(x_val) =", len(x_val))
print("\tlen(partial_x_train) =", len(partial_x_train))
print("\tlen(y_val) =", len(y_val))
print("\tlen(partial_y_train) =", len(partial_y_train))

print("* Code 3-8 训练模型")
# epochs = 9 就够了，随着训练迭代次数增多，训练集的精度越来越高，测试集的精度开始下降，说明训练集过拟合
# verbose: Verbosity mode, 0 (silent), 1 (verbose), 2 (semi-verbose)
model.fit(partial_x_train, partial_y_train, epochs = 9, batch_size = 512,
          validation_data = (x_val, y_val), use_multiprocessing = True, verbose = 2)
print("\t模型预测-->", end = '')
results = model.evaluate(x_test, one_hot_test_labels, verbose = 0)
print("\t损失值 = {}，精确度 = {}".format(results[0], results[1]))
predictions = model.predict(x_test)
print("\t前10个预测结果中最大值所在列 =")
print(np.argmax(predictions[:10], 1))
print("\t模型预测，前10个预测结果 =")
print(predictions[:10])

history = model.fit(partial_x_train, partial_y_train, epochs = 20, batch_size = 512,
                    validation_data = (x_val, y_val), use_multiprocessing = True, verbose = 2)
predictions = model.predict(x_test)
print("\t前10个预测结果中最大值所在列 =")
print(np.argmax(predictions[:10], 1))
print("\t前10个预测结果（每条数据有46个分类概率，最大概率为最可能的分类） =")
print(predictions[:10])

history_dict = history.history
print("\thistory_dict.keys() =", history_dict.keys())
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, len(loss_values) + 1)

plt.figure()
plt.plot(epochs, loss_values, 'bo', label = '训练集的损失')  # bo 蓝色圆点
plt.plot(epochs, val_loss_values, 'b', label = '验证集的损失')  # b 蓝色实线
plt.title('图3-7：训练损失和验证损失')
plt.xlabel('Epochs--批次')
plt.ylabel('Loss--损失')
plt.legend()

acc = history_dict['binary_accuracy']
val_acc = history_dict['val_binary_accuracy']

plt.figure()
plt.plot(epochs, acc, 'bo', label = '训练集的精度')
plt.plot(epochs, val_acc, 'b', label = '验证集的精度')
plt.title('图3-8：训练精度和验证精度')
plt.xlabel('Epochs--批次')
plt.ylabel('Accuracy--精度')
plt.ylim([0., 1.2])
plt.legend()

# 标签采用整数编码与采用 One-Hot 编码的区别
# 本质是相同的
# 损失函数由 categorical_crossentropy 换成 sparse_categorical_crossentropy
y_train = np.array(train_labels)
y_test = np.array(test_labels)
model.compile(optimizer = keras.optimizers.rmsprop(lr = 0.001),
              loss=keras.losses.sparse_categorical_crossentropy,
              metrics = [keras.metrics.binary_accuracy])
# 运行结束的提醒
winsound.Beep(600, 500)
if len(plt.get_fignums()) != 0:
    plt.show()
pass
