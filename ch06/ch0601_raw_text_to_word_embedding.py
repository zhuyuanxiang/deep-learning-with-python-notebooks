# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   deep-learning-with-python-notebooks
@File       :   ch0601_raw_text_to_word_embedding.py
@Version    :   v0.1
@Time       :   2019-11-23 14:26
@License    :   (C)Copyright 2018-2019, zYx.Tom
@Reference  :   《Python 深度学习，Francois Chollet》, Sec060103，P155
@Desc       :   深度学习用于文本和序列，处理文本数据——从原始文本到词嵌入
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np  # pip install numpy<1.17，小于1.17就不会报错
import winsound
from keras.activations import relu, sigmoid
from keras.layers import Dense, Flatten
from keras.layers import Embedding
from keras.losses import binary_crossentropy
from keras.models import Sequential
from keras.optimizers import rmsprop
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

from tools import plot_classes_results

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

# Listing 6.8 准备 IMDB 原始数据的标签
imdb_dir = "C:/Users/Administrator/PycharmProjects/Data/aclImdb/"
train_dir = os.path.join(imdb_dir, 'train')
test_dir = os.path.join(imdb_dir, 'test')

# Listing 6.9 对 IMDB 原始数据的文本进行分词
max_len = 100  # 每条评论的最大长度
max_words = 10000  # 单词表大小
epochs = 10
batch_size = 32
training_samples = 200
validation_samples = 10000


def load_data_set(data_dir):
    labels, texts = [], []
    for label_type in ['neg', 'pos']:
        dir_name = os.path.join(data_dir, label_type)
        for fname in os.listdir(dir_name):
            # print(fname)
            if fname[-4:] == '.txt':
                f = open(os.path.join(dir_name, fname), encoding = 'utf-8')
                texts.append(f.read())
                f.close()
                if label_type == 'neg':
                    labels.append(0)
                else:
                    labels.append(1)
                    pass
                pass
            pass
        pass
    return texts, labels


train_texts, train_labels = load_data_set(train_dir)
test_texts, test_labels = load_data_set(test_dir)

tokenizer = Tokenizer(num_words = max_words)
tokenizer.fit_on_texts(train_texts)
word_index = tokenizer.word_index
print("Found {} unique tokens.".format(len(word_index)))

train_sequences = tokenizer.texts_to_sequences(train_texts)
train_data = pad_sequences(train_sequences, maxlen = max_len)
train_labels = np.asarray(train_labels)
print("Shape of data tensor:", train_data.shape)
print("Shape of label tensor:", train_labels.shape)

indices = np.arange(train_data.shape[0])
np.random.shuffle(indices)
train_data = train_data[indices]
train_labels = train_labels[indices]

x_train = train_data[:training_samples]
y_train = train_labels[:training_samples]
x_val = train_data[training_samples:training_samples + validation_samples]
y_val = train_labels[training_samples:training_samples + validation_samples]

test_sequences = tokenizer.texts_to_sequences(test_texts)
x_test = pad_sequences(test_sequences, maxlen = max_len)
y_test = np.asarray(test_labels)

# downloading...https://nlp.stanford.edu/projects/glove/
# 下载：http://nlp.stanford.edu/data/glove.6B.zip
# Listing 6.10 解析 GloVe 词嵌入文件
glove_dir = "C:/Users/Administrator/PycharmProjects/Data/GloVe/"
embeddings_index = {}
f = open(os.path.join(glove_dir, 'glove.6B.200d.txt'), encoding = 'utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype = 'float32')
    embeddings_index[word] = coefs
    pass
f.close()
print("Found {} word vectors.".format(len(embeddings_index)))

# Listing 6.11 准备 GloVe 词嵌入矩阵
embedding_dim = 200
embedding_matrix = np.zeros((max_words, embedding_dim))
for word, i in word_index.items():
    if i < max_words:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            pass
        pass
    pass


def train_model(model):
    # Listing 6.14 训练与评估模型
    # 这个声明放在最前面会报错
    from keras.metrics import binary_accuracy
    model.compile(optimizer = rmsprop(lr = 0.001), loss = binary_crossentropy, metrics = [binary_accuracy])
    return model.fit(x_train, y_train, epochs = epochs, batch_size = batch_size, validation_data = (x_val, y_val),
                     verbose = 2, use_multiprocessing = True)


# ----------------------------------------------------------------------
def definite_model_use_GloVe():
    # Listing 6.12 模型定义
    title = "使用预训练词嵌入"
    model = Sequential()
    model.add(Embedding(max_words, embedding_dim, input_length = max_len))
    model.add(Flatten())
    model.add(Dense(32, activation = relu))
    model.add(Dense(1, activation = sigmoid))
    # Listing 6.13 将 GloVe 训练好的词嵌入加载到 Embedding 层中
    model.layers[0].set_weights([embedding_matrix])
    model.layers[0].trainable = False
    model.summary()
    history = train_model(model)
    plot_classes_results(history, title, epochs)
    model.save_weights('pre_trained_glove_model.h5')
    model.load_weights('pre_trained_glove_model.h5')
    print(title + "评估测试集", model.evaluate(x_test, y_test))


# ----------------------------------------------------------------------
def definite_model_no_GloVe():
    title = "不使用预训练词嵌入"
    model = Sequential()
    model.add(Embedding(max_words, embedding_dim, input_length = max_len))
    model.add(Flatten())
    model.add(Dense(32, activation = relu))
    model.add(Dense(1, activation = sigmoid))
    model.summary()
    history = train_model(model)
    plot_classes_results(history, title, epochs)
    print(title + "评估测试集", model.evaluate(x_test, y_test))


# ----------------------------------------------------------------------
definite_model_use_GloVe()
definite_model_no_GloVe()

# 运行结束的提醒
winsound.Beep(600, 500)
if len(plt.get_fignums()) != 0:
    plt.show()
pass
