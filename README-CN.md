# 《Python 深度学习》

[美国]弗朗索瓦.肖莱 著；张亮 译

# 环境准备

## 开发环境
- 开发环境
    - Python 3.6.7
    - Anaconda 1.9.2（建议不要升级到1.9.5，会出错）
    - 开发包(必须）
        - Keras 2.2.5
        - TensorFlow 1.3.0
        - NumPy 1.16.5
        - MatplotLib 3.1.2
            - matplotlib绘图需要提供中文字体支持。
                1. 将 “\Tools” 目录下的 “YaHei.Consolas.1.12.ttf” 文件拷贝到 “\Lib\site-packages\matplotlib\mpl-data\fonts\ttf” 目录下。
                2. 将 “\Tools” 目录下的 “matplotlibrc” 文件拷贝到 “\Lib\site-packages\matplotlib\mpl-data\” 目录下，
                拷贝过程中可以直接覆盖原始文件，也可以将原始文件改名。
        - Scikit-Learn 0.21.3
        - NLTK 3.4.4
        - Theano 1.0.4
        - m2w-toolchain 5.3.0
        - libpython 2.0
    - 开发包（可选）（我的是旧的显卡 820M）
        - CUDA 7.5
        - cuDNN 不支持
    - IDE
        - PyCharm Community Edition：免费，够用，不支持Jupiter（可以用VS Code代替）
        - PyCharm Professional Edition：支持的功能更多，需要有个大屏幕

# 补充说明

**没有Jupiter可以访问下面的链接**
* Chapter 2:
    * [2.1: 神经网络初探](http://nbviewer.jupyter.org/github/fchollet/deep-learning-with-python-notebooks/blob/master/2.1-a-first-look-at-a-neural-network.ipynb)
* Chapter 3:
    * [3.5: Classifying movie reviews](http://nbviewer.jupyter.org/github/fchollet/deep-learning-with-python-notebooks/blob/master/3.5-classifying-movie-reviews.ipynb)
    * [3.6: Classifying newswires](http://nbviewer.jupyter.org/github/fchollet/deep-learning-with-python-notebooks/blob/master/3.6-classifying-newswires.ipynb)
    * [3.7: Predicting house prices](http://nbviewer.jupyter.org/github/fchollet/deep-learning-with-python-notebooks/blob/master/3.7-predicting-house-prices.ipynb)
* Chapter 4:
    * [4.4: Underfitting and overfitting](http://nbviewer.jupyter.org/github/fchollet/deep-learning-with-python-notebooks/blob/master/4.4-overfitting-and-underfitting.ipynb)
* Chapter 5:
    * [5.1: Introduction to convnets](http://nbviewer.jupyter.org/github/fchollet/deep-learning-with-python-notebooks/blob/master/5.1-introduction-to-convnets.ipynb)
    * [5.2: Using convnets with small datasets](http://nbviewer.jupyter.org/github/fchollet/deep-learning-with-python-notebooks/blob/master/5.2-using-convnets-with-small-datasets.ipynb)
    * [5.3: Using a pre-trained convnet](http://nbviewer.jupyter.org/github/fchollet/deep-learning-with-python-notebooks/blob/master/5.3-using-a-pretrained-convnet.ipynb)
    * [5.4: Visualizing what convnets learn](http://nbviewer.jupyter.org/github/fchollet/deep-learning-with-python-notebooks/blob/master/5.4-visualizing-what-convnets-learn.ipynb)
* Chapter 6:
    * [6.1: One-hot encoding of words or characters](http://nbviewer.jupyter.org/github/fchollet/deep-learning-with-python-notebooks/blob/master/6.1-one-hot-encoding-of-words-or-characters.ipynb)
    * [6.1: Using word embeddings](http://nbviewer.jupyter.org/github/fchollet/deep-learning-with-python-notebooks/blob/master/6.1-using-word-embeddings.ipynb)
    * [6.2: Understanding RNNs](http://nbviewer.jupyter.org/github/fchollet/deep-learning-with-python-notebooks/blob/master/6.2-understanding-recurrent-neural-networks.ipynb)
    * [6.3: Advanced usage of RNNs](http://nbviewer.jupyter.org/github/fchollet/deep-learning-with-python-notebooks/blob/master/6.3-advanced-usage-of-recurrent-neural-networks.ipynb)
    * [6.4: Sequence processing with convnets](http://nbviewer.jupyter.org/github/fchollet/deep-learning-with-python-notebooks/blob/master/6.4-sequence-processing-with-convnets.ipynb)
* Chapter 8:
    * [8.1: Text generation with LSTM](http://nbviewer.jupyter.org/github/fchollet/deep-learning-with-python-notebooks/blob/master/8.1-text-generation-with-lstm.ipynb)
    * [8.2: Deep dream](http://nbviewer.jupyter.org/github/fchollet/deep-learning-with-python-notebooks/blob/master/8.2-deep-dream.ipynb)
    * [8.3: Neural style transfer](http://nbviewer.jupyter.org/github/fchollet/deep-learning-with-python-notebooks/blob/master/8.3-neural-style-transfer.ipynb)
    * [8.4: Generating images with VAEs](http://nbviewer.jupyter.org/github/fchollet/deep-learning-with-python-notebooks/blob/master/8.4-generating-images-with-vaes.ipynb)
    * [8.5: Introduction to GANs](http://nbviewer.jupyter.org/github/fchollet/deep-learning-with-python-notebooks/blob/master/8.5-introduction-to-gans.ipynb)


# Ch01 什么是深度学习

- 基本概念
- 发展历史
- 未来趋势

