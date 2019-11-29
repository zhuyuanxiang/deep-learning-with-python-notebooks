# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   deep-learning-with-python-notebooks
@File       :   ch0801_text_generation_with_lstm.py
@Version    :   v0.1
@Time       :   2019-11-28 8:12
@License    :   (C)Copyright 2018-2019, zYx.Tom
@Reference  :   《Python 深度学习，Francois Chollet》, Sec0801，P231
@Desc       :   生成式深度学习，使用 LSTM 生成文本
"""
import os
import random
import sys

import matplotlib.pyplot as plt
import numpy as np  # pip install numpy<1.17，小于1.17就不会报错
import winsound
from keras.activations import softmax
from keras.layers import Dense
from keras.layers import LSTM
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
def reweight_distribution(original_distribution, temperature = 0.5):
    """

    :param original_distribution: 概率值组成的一维 Numpy 数组，数据之和等于1
    :param temperature: 用于定量描述输出分布的熵的因子
    :return: 将加权生的分布归一化，保证数组之和为1
    """
    distribution = np.exp(np.log(original_distribution) / temperature)
    return distribution / np.sum(distribution)


# ----------------------------------------------------------------------
print("Listing 8.2：加载数据...")
data_dir = "C:/Users/Administrator/PycharmProjects/Data/"
# fname = os.path.join(data_dir, 'nietzsche.txt')
# fname = os.path.join(data_dir, 'shakespeare-macbeth.txt')
# fname = os.path.join(data_dir, 'Chinese_Mandarin-UTF8.txt')
fname = os.path.join(data_dir, 'jinggangjing-CN-UTF8.txt')
f = open(fname, encoding = 'utf-8')
text = f.read().lower()
print("Corpus length:", len(text))
f.close()
# ----------------------------------------------------------------------
max_len = 60  # 每个序列由 60 个字符组成
step = 3  # 每隔3个字符采样下一个新的序列
sentences = []  # 保存所提取的序列
next_chars = []  # 保存目标（即序列所对应的下一个字符）

for i in range(0, len(text) - max_len, step):
    sentences.append(text[i:i + max_len])
    next_chars.append(text[i + max_len])
    pass
print("Number of sequences:", len(sentences))
# 语料中唯一字符组成的列表
chars = sorted(list(set(text)))
print("Unique characters:", len(chars))
# 一个字典，将唯一字符映射为它在列表 chars 中的索引
char_indices = dict((char, chars.index(char)) for char in chars)
# 将字符 基于 One-Hot 编码为二进制数组
print("Vectorization...")
x = np.zeros((len(sentences), max_len, len(chars)), dtype = np.bool)
y = np.zeros((len(sentences), len(chars)), dtype = np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
        pass
    y[i, char_indices[next_chars[i]]] = 1
    pass

# ----------------------------------------------------------------------
print("Listing 8.4 用于预测下一个字符的单层 LSTM 模型")
model = Sequential()
model.add(LSTM(128, input_shape = (max_len, len(chars))))
model.add(Dense(len(chars), activation = softmax))
model.summary()
model.compile(optimizer = rmsprop(lr = 0.01), loss = categorical_crossentropy)


# ----------------------------------------------------------------------
# print("Listing 8.6 给定模型的预测，采样下一个字符")
def sample(preds, temperature = 1.0):
    # 理解可以参考前面的 reweight_distribution() 函数
    preds = np.asarray(preds).astype('float64')
    preds = np.exp(np.log(preds) / temperature)
    preds = preds / np.sum(preds)
    # 基于多项式分布抽取最可能的字符
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


# ----------------------------------------------------------------------
print("Listing 8.7 循环生成文本")
for epoch in range(1, 60):
    print('epoch:', epoch)

    # 每次循环都将模型重新拟合一次
    model.fit(x, y, batch_size = 128, epochs = 1, verbose = 2, use_multiprocessing = True)

    # 随机选择一个文本种子
    start_index = random.randint(0, len(text) - max_len - 1)
    generated_seed = text[start_index:start_index + max_len]  # 原代码这里有误
    print("--- Generating with seed: “" + generated_seed + '”')

    # 尝试不同的采样温度
    for temperature in [0.2, 0.5, 1.0, 1.2]:
        print("------ temperature:", temperature)
        generated_text = generated_seed  # 种子是不能变化的，否则无法对比
        sys.stdout.write(generated_text)  # 先输出种子文本
        for i in range(400):
            # 先将种子文本进行 One-Hot 编码
            sampled = np.zeros((1, max_len, len(chars)))
            for t, char in enumerate(generated_text):
                sampled[0, t, char_indices[char]] = 1.
                pass

            # 对下一个字符进行采样
            preds = model.predict(sampled, verbose = 0)[0]
            next_index = sample(preds, temperature)
            next_char = chars[next_index]

            # generated_text += next_char
            generated_text = generated_text[1:] + next_char
            sys.stdout.write(next_char)
            sys.stdout.flush()
            pass
        print()
        print('=' * 50)
        pass
    print('*' * 50)
    pass

# ----------------------------------------------------------------------
# 运行结束的提醒
winsound.Beep(600, 500)
if len(plt.get_fignums()) != 0:
    plt.show()
pass
