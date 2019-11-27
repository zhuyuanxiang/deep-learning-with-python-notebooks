# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   deep-learning-with-python-notebooks
@File       :   ch0701_keras_functional_api.py
@Version    :   v0.1
@Time       :   2019-11-26 17:27
@License    :   (C)Copyright 2018-2019, zYx.Tom
@Reference  :   《Python 深度学习，Francois Chollet》, Sec0701，P196
@Desc       :   高级的深度学习最佳实践，使用Keras Functional API
"""
import os
import sys

import keras
import matplotlib.pyplot as plt
import numpy as np  # pip install numpy<1.17，小于1.17就不会报错
import winsound
from keras import Input, Model
from keras.activations import relu, sigmoid, softmax
from keras.applications import Xception
from keras.layers import (AveragePooling2D, concatenate, Conv1D, Conv2D, Embedding, GlobalMaxPooling1D, LSTM,
                          MaxPooling1D, MaxPooling2D, )
from keras.layers import Dense
from keras.losses import binary_crossentropy, categorical_crossentropy, mse
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
epochs = 10
batch_size = 128
verbose = 2


# ----------------------------------------------------------------------
def sequential_realize():
    print("使用 Keras Functional API 使用 Sequential")
    x_train = np.random.random((1000, 64))
    y_train = np.random.random((1000, 10))
    seq_title = "使用Sequential实现的模型"
    seq_model = Sequential(name = seq_title)
    seq_model.add(Dense(32, activation = relu, input_shape = (64,)))
    seq_model.add(Dense(32, activation = relu))
    seq_model.add(Dense(10, activation = softmax))
    seq_model.summary()
    seq_model.compile(optimizer = rmsprop(), loss = categorical_crossentropy)
    seq_model.fit(x_train, y_train, epochs = epochs, batch_size = batch_size,
                  verbose = verbose, use_multiprocessing = True)
    print(seq_title + ":", seq_model.evaluate(x_train, y_train))

    api_title = "使用 Functional API 实现的模型"
    input_tensor = Input(shape = (64,))
    x = Dense(32, activation = relu)(input_tensor)
    x = Dense(32, activation = relu)(x)
    output_tensor = Dense(10, activation = softmax)(x)
    api_model = Model(input_tensor, output_tensor, name = api_title)
    api_model.summary()
    api_model.compile(optimizer = rmsprop(), loss = categorical_crossentropy)
    api_model.fit(x_train, y_train, epochs = epochs, batch_size = batch_size)
    print(api_title + ":", api_model.evaluate(x_train, y_train))


def sequential_realize_wrong():
    print("不相关的输入与输出连接")
    input_tensor = Input(shape = (64,))
    x = Dense(32, activation = relu)(input_tensor)
    x = Dense(32, activation = relu)(x)
    output_tensor = Dense(10, activation = softmax)(x)
    unrelated_input = Input(shape = (32,))
    bad_model = Model(unrelated_input, output_tensor)
    pass


def multi_input_realize():
    print("Listing 7.1 使用 Functional API 实现的双输入问答模型")
    text_vocabulary_size = 10000
    question_vocabulary_size = 10000
    answer_vocabulary_size = 500

    num_samples = 10000
    max_length = 100
    epochs = 20
    text = np.random.randint(1, text_vocabulary_size, size = (num_samples, max_length))
    question = np.random.randint(1, question_vocabulary_size, size = (num_samples, max_length))
    answers = np.random.randint(answer_vocabulary_size, size = (num_samples,))
    answers = keras.utils.to_categorical(answers, answer_vocabulary_size)

    text_input = Input(shape = (None,), dtype = 'int32', name = 'text')
    embedded_text = Embedding(text_vocabulary_size, 64)(text_input)
    encoded_text = LSTM(32)(embedded_text)

    question_input = Input(shape = (None,), dtype = 'int32', name = 'question')
    embedded_question = Embedding(question_vocabulary_size, 32)(question_input)
    encoded_question = LSTM(16)(embedded_question)

    concatenated = concatenate([encoded_text, encoded_question], axis = -1)
    output = Dense(answer_vocabulary_size, activation = softmax)(concatenated)
    model = Model([text_input, question_input], output, name = "使用 Functional API 实现的双输入问答模型")
    model.summary()
    model.compile(optimizer = rmsprop(), loss = categorical_crossentropy, metrics = ['acc'])
    # 使用输入组成的列表来输入数据
    # model.fit([text, question], answers, epochs = epochs, batch_size = batch_size,
    # verbose = verbose,use_multiprocessing = True)
    # 使用输入组成的字典来输入数据
    model.fit({'text': text, 'question': question}, answers, epochs = epochs, batch_size = batch_size,
              verbose = verbose, use_multiprocessing = True)


def multi_output_realize():
    vocabulary_size = 50000
    num_income_groups = 10
    posts_input = Input(shape = (None,), dtype = 'int32', name = 'posts')
    embedded_posts = Embedding(256, vocabulary_size)(posts_input)
    x = Conv1D(128, 5, activation = relu)(embedded_posts)
    x = MaxPooling1D(5)(x)
    x = Conv1D(256, 6, activation = relu)(x)
    x = Conv1D(256, 6, activation = relu)(x)
    x = MaxPooling1D(5)(x)
    x = Conv1D(256, 6, activation = relu)(x)
    x = Conv1D(256, 6, activation = relu)(x)
    x = GlobalMaxPooling1D()(x)
    x = Dense(128, activation = relu)(x)

    age_prediction = Dense(1, name = 'age')(x)
    income_prediction = Dense(num_income_groups, activation = softmax, name = 'income')(x)
    gender_prediction = Dense(1, activation = sigmoid, name = 'gender')(x)

    model = Model(posts_input, [age_prediction, income_prediction, gender_prediction])
    model.compile(optimizer = rmsprop(), loss = [mse, categorical_crossentropy, binary_crossentropy])
    # 使用name定义损失函数
    model.compile(optimizer = rmsprop(), loss = {
            'age': mse, 'income': categorical_crossentropy, 'gender': binary_crossentropy
    })
    # Listing 7.5：多输出模型的损失加权
    model.compile(optimizer = rmsprop(), loss = {
            'age': mse, 'income': categorical_crossentropy, 'gender': binary_crossentropy
    }, loss_weights = {'age': 0.25, 'income': 1., 'gender': 10.})

    model.fit(posts, [age_targets, income_targets, gender_targets], epochs = epochs, batch_size = batch_size)
    # 使用name定义标签
    model.fit(posts, {
            'age': age_targets, 'income': income_targets, 'gender': gender_targets
    }, epochs = epochs, batch_size = batch_size)


def directed_acyclic_graphs_realize():
    print("图7-8：Inception模块的实现")
    input_data = Input(shape = (None,))
    branch_a = Conv2D(128, 1, activation = relu, strides = 2)(input_data)

    branch_b = Conv2D(128, 1, activation = relu)(input_data)
    branch_b = Conv2D(128, 2, activation = relu, strides = 2)(branch_b)

    branch_c = AveragePooling2D(3, strides = 2)(input_data)
    branch_c = Conv2D(128, 3, activation = relu)(branch_c)

    branch_d = Conv2D(128, 1, activation = relu)(input_data)
    branch_d = Conv2D(128, 3, activation = relu)(branch_d)
    branch_d = Conv2D(128, 3, activation = relu, strides = 2)(branch_d)

    output = concatenate([branch_a, branch_b, branch_c, branch_d], axis = -1)


def resnet_realize():
    input_data = Input(shape = (None,))

    # 恒等残差连接（Identity Residual Connection）
    # 两个数据维度相同，直接相加
    y = Conv2D(128, 3, activation = relu, padding = 'same')(input_data)
    y = Conv2D(128, 3, activation = relu, padding = 'same')(y)
    y = Conv2D(128, 3, activation = relu, padding = 'same')(y)
    y = keras.layers.add([y, input_data])

    # 线性残差连接（Linear Residual Connection）
    # 两个数据维度不同，先使用卷积将其中一个数据整理成与另一个数据相同的形状
    y = Conv2D(128, 3, activation = relu, padding = 'same')(input_data)
    y = Conv2D(128, 3, activation = relu, padding = 'same')(y)
    y = MaxPooling2D(2, strides = 2)(y)
    residual = Conv2D(128, 1, strides = 2, padding = 'same')(input_data)
    y = keras.layers.add([y, residual])


# ToSee：同时基于两组数据进行学习，那么两组数据相互之间是否应该有个数学关系（例如：相加、加权、或者轮流输入）
def layer_weight_sharing_realize():
    # 将一个 LSTM 层实例化一次
    lstm = LSTM(32)

    # 构建模型的左分支，输入是长度为 128 的向量组成的变长序列
    left_input = Input(shape = (None, 128))
    left_output = lstm(left_input)

    # 构建模型的右分支，如果调用已经存在的层实例，那么就会重复使用它的权重
    right_input = Input(shape = (None, 128))
    right_output = lstm(right_input)

    # 在上面构建一个分类器
    merged = concatenate([left_output, right_output], axis = -1)
    predictions = Dense(1, activation = sigmoid)(merged)

    # 将模型实例化并且训练，训练这种模型时，基于两个输入对LSTM层的权重进行更新
    model = Model([left_input, right_input], predictions)
    model.summary()
    # model.fit([left_data, right_data], targets)


def models_as_layers_realize():
    # 图像处理基础模型是 Xception 网络（只包括卷积基）
    xception_base = Xception(weights = None, include_top = False)
    # 输入是 250 x 250 的 RGB 图像
    left_input = Input(shape = (250, 250, 3))
    right_input = Input(shape = (250, 250, 3))

    # 对相同的视觉模型调用两次
    left_features = xception_base(left_input)
    right_features = xception_base(right_input)

    # 合并后的特征包含来自左右两个视觉输入中的信息
    merged_features = concatenate([left_features, right_features], axis = -1)


# ----------------------------------------------------------------------
# print('*' * 50)
# sequential_realize()
# print('*' * 50)
# sequential_realize_wrong()
# print('*' * 50)
# multi_input_realize()
print('*' * 50)
layer_weight_sharing_realize()
# ----------------------------------------------------------------------
# 运行结束的提醒
winsound.Beep(600, 500)
if len(plt.get_fignums()) != 0:
    plt.show()
pass
