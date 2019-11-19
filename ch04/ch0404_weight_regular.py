# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   deep-learning-with-python-notebooks
@File       :   ch0404_weight_regular.py
@Version    :   v0.1
@Time       :   2019-11-19 15:05
@License    :   (C)Copyright 2018-2019, zYx.Tom
@Reference  :   《Python 深度学习，Francois Chollet》, Sec0404，P83
@Desc       :   机器学习基础，过拟合与欠拟合
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
from keras import regularizers
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

# 参数设置
epochs = 20


def get_original_model():
    print("构造原始模型")
    model = models.Sequential()
    model.add(layers.Dense(16, activation = activations.relu, input_shape = (10000,)))
    model.add(layers.Dense(16, activation = activations.relu))
    model.add(layers.Dense(1, activation = activations.sigmoid))
    return model


def get_smaller_model():
    print("构造更小模型")
    model = models.Sequential()
    model.add(layers.Dense(4, activation = activations.relu, input_shape = (10000,)))
    model.add(layers.Dense(4, activation = activations.relu))
    model.add(layers.Dense(1, activation = activations.sigmoid))
    return model


def get_bigger_model():
    print("构造更大模型")
    model = models.Sequential()
    model.add(layers.Dense(512, activation = activations.relu, input_shape = (10000,)))
    model.add(layers.Dense(512, activation = activations.relu))
    model.add(layers.Dense(1, activation = activations.sigmoid))
    return model


def get_l2_regular_model():
    print("构造L2正则模型")
    model = models.Sequential()
    model.add(layers.Dense(16, activation = activations.relu, input_shape = (10000,),
                           kernel_regularizer = regularizers.l2(0.001)))
    model.add(layers.Dense(16, activation = activations.relu, kernel_regularizer = regularizers.l2(0.001)))
    model.add(layers.Dense(1, activation = activations.sigmoid))
    return model


def get_l1_regular_model():
    print("构造L1正则模型")
    model = models.Sequential()
    model.add(layers.Dense(16, activation = activations.relu, input_shape = (10000,),
                           kernel_regularizer = regularizers.l1(0.001)))
    model.add(layers.Dense(16, activation = activations.relu, kernel_regularizer = regularizers.l1(0.001)))
    model.add(layers.Dense(1, activation = activations.sigmoid))
    return model


def get_l1_l2_regular_model():
    print("构造L1 & L2正则模型")
    model = models.Sequential()
    model.add(layers.Dense(16, activation = activations.relu, input_shape = (10000,),
                           kernel_regularizer = regularizers.l1_l2(l1=0.001,l2=0.001)))
    model.add(layers.Dense(16, activation = activations.relu, kernel_regularizer = regularizers.l1_l2(0.001)))
    model.add(layers.Dense(1, activation = activations.sigmoid))
    return model


def get_dropout_model():
    print("构造dropout模型")
    model = models.Sequential()
    model.add(layers.Dense(16, activation = activations.relu, input_shape = (10000,)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(16, activation = activations.relu))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation = activations.sigmoid))
    return model


def compile_model(model):
    print("* Code 3-4 编译模型")
    model.compile(optimizer = optimizers.RMSprop(lr = 0.001),
                  loss = losses.binary_crossentropy,
                  metrics = [metrics.binary_accuracy])
    return model


def fit_evaluate_model(model, x_train_data, y_train_data, x_val_data, y_val_data, x_test_data, y_test_data):
    print("* Code 3-8 训练模型")
    history = model.fit(x_train_data, y_train_data, epochs = epochs, batch_size = 512,
                        validation_data = (x_val_data, y_val_data), use_multiprocessing = True, verbose = 2)
    return history.history['val_loss']


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

# 3.4.4 验证你的方法
print("* Code 3-7 留出验证集")
x_val = x_train[:10000]
partial_x_train = x_train[10000:]

y_val = y_train[:10000]
partial_y_train = y_train[10000:]

# 3.4.3 构建网络
print("* Code 3-3 模式定义")
epochs_range = range(1, epochs + 1)
original_model = compile_model(get_original_model())
original_val_loss_values = fit_evaluate_model(
        original_model, partial_x_train, partial_y_train, x_val, y_val, x_test, y_test
)
plt.plot(epochs_range, original_val_loss_values, 'bo', label = '原始模型')  # bo 蓝色圆点

smaller_model = compile_model(get_smaller_model())
smaller_val_loss_values = fit_evaluate_model(
        smaller_model, partial_x_train, partial_y_train, x_val, y_val, x_test, y_test
)
plt.plot(epochs_range, smaller_val_loss_values, 'g^', label = '更小的模型')

bigger_model = compile_model(get_bigger_model())
bigger_val_loss_values = fit_evaluate_model(
        bigger_model, partial_x_train, partial_y_train, x_val, y_val, x_test, y_test
)
plt.plot(epochs_range, bigger_val_loss_values, 'rv', label = '更大的模型')

l2_regular_model = compile_model(get_l2_regular_model())
l2_regular_val_loss_values = fit_evaluate_model(
        l2_regular_model, partial_x_train, partial_y_train, x_val, y_val, x_test, y_test
)
plt.plot(epochs_range, l2_regular_val_loss_values, 'c-', label = 'l2正则化模型')

l1_regular_model = compile_model(get_l1_regular_model())
l1_regular_val_loss_values = fit_evaluate_model(
        l1_regular_model, partial_x_train, partial_y_train, x_val, y_val, x_test, y_test
)
plt.plot(epochs_range, l1_regular_val_loss_values, 'm--', label = 'l1正则化模型')

l1_l2_regular_model = compile_model(get_l1_l2_regular_model())
l1_l2_regular_val_loss_values = fit_evaluate_model(
        l1_l2_regular_model, partial_x_train, partial_y_train, x_val, y_val, x_test, y_test
)
plt.plot(epochs_range, l1_l2_regular_val_loss_values, 'y-', label = 'l1_l2正则化模型')

dropout_model = compile_model(get_dropout_model())
dropout_val_loss_values = fit_evaluate_model(
        dropout_model, partial_x_train, partial_y_train, x_val, y_val, x_test, y_test
)
plt.plot(epochs_range, dropout_val_loss_values, 'k*', label = 'dropout正则化模型')

plt.title('图3-7：不同模型的验证损失')
plt.xlabel('Epochs--批次')
plt.ylabel('Validation Loss--验证损失')
plt.legend()

# 运行结束的提醒
winsound.Beep(600, 500)
if len(plt.get_fignums()) != 0:
    plt.show()
pass
