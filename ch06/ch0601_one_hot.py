# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   deep-learning-with-python-notebooks
@File       :   ch0601_one_hot.py
@Version    :   v0.1
@Time       :   2019-11-23 10:21
@License    :   (C)Copyright 2018-2019, zYx.Tom
@Reference  :   《Python 深度学习，Francois Chollet》, Sec0601，P147
@Desc       :   深度学习用于文本和序列，处理文本数据——One-Hot编码
"""
import os
import string
import sys

import matplotlib.pyplot as plt
import numpy as np  # pip install numpy<1.17，小于1.17就不会报错
import winsound
from keras_preprocessing.text import Tokenizer

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

samples = ['The cat sat on the mat.', 'The dog ate my homework.']


# Listing 6.1 单词级别的 One-Hot 编码（简单示例）
def one_hot_word():
    token_index = {}
    for sample in samples:
        for word in sample.split():
            if word not in token_index:
                # 索引的长度len(token_index)
                token_index[word] = len(token_index) + 1
                pass
            pass
        pass
    max_length = 10  # 每句话的单词个数
    results = np.zeros(shape = (len(samples), max_length, max(token_index.values()) + 1))
    for i, sample in enumerate(samples):
        for j, word in list(enumerate(sample.split()))[:max_length]:
            index = token_index.get(word)
            results[i, j, index] = 1.
            pass
        pass


# Listing 6.2 字符级别的 One-Hot 编码（简单示例）
def one_hot_character():
    characters = string.printable  # 所有可以打印的 ASCII字符
    token_index = dict(zip(characters, range(1, len(characters) + 1)))  # 原代码这里反了

    max_length = 50  # 每句话的字符个数
    results = np.zeros((len(samples), max_length, len(token_index) + 1))  # 原代码因为上面的修改而需要变动
    for i, sample in enumerate(samples):
        for j, character in enumerate(sample):
            index = token_index.get(character)
            results[i, j, index] = 1.
            pass
        pass


# Listing 6.3 用 Keras 实现单词级别的 One-Hot 编码
# 编码结果与前面的 One-Hot 编码的结果不一样！
# 每个句子中的所有单词被编码在一个向量中，如果需要单独编码可以把句子中的单词单独拆分出来即可。
def one_hot_word_with_keras():
    tokenizer = Tokenizer(num_words = 1000)
    tokenizer.fit_on_texts(samples)
    sequences = tokenizer.texts_to_sequences(samples)
    one_hot_results = tokenizer.texts_to_matrix(samples, mode = 'binary')
    word_index = tokenizer.word_index
    print("Found {} unique tokens.".format(len(word_index)))


# Listing 6.4 使用散列技巧的单词级别的 One-Hot 编码（简单示例）
# 单词向量的维度是1000，如果单词数量接近甚至超过1000，那么散列冲突的频率就会很高，严重影响编码的效率和质量
def one_hot_word_with_hash_set():
    dimensionality = 1000
    max_length = 10
    results = np.zeros((len(samples), max_length, dimensionality))
    for i, sample in enumerate(samples):
        for j, word in list(enumerate(sample.split()))[:max_length]:
            # 通过下面的散列函数给出单词的索引
            index = abs(hash(word)) % dimensionality
            results[i, j, index] = 1.


if __name__ == '__main__':
    # one_hot_word()
    # one_hot_character()
    # one_hot_word_with_keras()
    one_hot_word_with_hash_set()
    # 运行结束的提醒
    winsound.Beep(600, 500)
    if len(plt.get_fignums()) != 0:
        plt.show()
    pass
