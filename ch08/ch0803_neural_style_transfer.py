# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   deep-learning-with-python-notebooks
@File       :   ch0803_neural_style_transfer.py
@Version    :   v0.1
@Time       :   2019-11-28 16:37
@License    :   (C)Copyright 2018-2019, zYx.Tom
@Reference  :   《Python 深度学习，Francois Chollet》, Sec0803，P241
@Desc       :   生成式深度学习，神经风格迁移
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np  # pip install numpy<1.17，小于1.17就不会报错
import winsound

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
print("Listing 8.14 定义初始化变量")
from keras.preprocessing.image import load_img, img_to_array

# 想要变换的图像的路径
target_image_path = 'C:/Users/Administrator/PycharmProjects/Data/Pictures/creative_commons_elephant.jpg'
# 风格图像的路径
style_reference_image_path = 'C:/Users/Administrator/PycharmProjects/Data/Pictures/popova.png'

# 需要生成的图像的尺寸
width, height = load_img(target_image_path).size
img_height = 400
img_width = int(width * img_height / height)
# ----------------------------------------------------------------------
print("Listing 8.15 辅助函数")
from keras.applications import vgg19


def preprocess_image(image_path):
    img = load_img(image_path, target_size = (img_height, img_width))
    img = img_to_array(img)
    img = np.expand_dims(img, axis = 0)
    # vgg19.preprocess_input()：
    # 减去 ImageNet 的平均像素值，使img的中心为0；
    # 将 图像由 RGB 格式 转换为 BGR 格式
    img = vgg19.preprocess_input(img)
    return img


def deprocess_image(x):
    # 以下操作相当于 vgg19.preprocess_input() 的逆操作
    # 加上 ImageNet 的平均像素值，使img恢复原来的分布
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # 将图像由 BGR 格式转换为 RGB 格式,
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x


# ----------------------------------------------------------------------
print("Listing 8.16 加载预训练的 VGG19 网络，并且将其应用于三张图像")
from keras import backend as K

target_image = K.constant(preprocess_image(target_image_path))
style_reference_image = K.constant(preprocess_image(style_reference_image_path))
# 占位符用于保存生成图像（占位符的知识来自于TensorFlow）
combination_image = K.placeholder((1, img_height, img_width, 3))

# theano 不支持
# 将三张图像合并为一个批量
input_tensor = K.concatenate([target_image, style_reference_image, combination_image], axis = 0)
# 利用三张图像组成的批量作为僌来构建 VGG19 网络
# 加载模型将使用预训练的 ImageNet 权重
model = vgg19.VGG19(input_tensor = input_tensor, weights = 'imagenet', include_top = False)
model.summary()
print('Model loaded.')


# ----------------------------------------------------------------------


def content_loss(base, combination):
    print("Listing 8.17 定义内容损失函数")
    return K.sum(K.square(combination - base))  # L2 范数


def gram_matrix(x):
    # 定义 Gram Matrix（格拉姆矩阵）
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram


def style_loss(style, combination):
    print("Listing 8.18 定义风格损失函数")
    channels = 3
    size = img_height * img_width
    return K.sum(K.square(gram_matrix(style) - gram_matrix(combination))) / (4. * (channels ** 2) * (size ** 2))


def total_variation_loss(x):
    print("Listing 8.19 定义总变差损失函数")
    a = K.square(x[:, :img_height - 1, :img_width - 1, :] - x[:, 1:, :img_width - 1, :])
    b = K.square(x[:, :img_height - 1, :img_width - 1, :] - x[:, :img_height - 1, 1:, :])
    return K.sum(K.pow(a + b, 1.25))


# ----------------------------------------------------------------------
print("Listing 8.20 定义需要最小化的最终损失")
# 将层的名称映射为激活张量的字典
outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])
# 用于内容损失的层
content_layer = 'block5_conv2'
# 用于风格损失的层
style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1', ]
total_variation_weight = 1e-4
style_weight = 1.
content_weight = 0.025
# 添加内容损失到总的损失中
# 在定义损失时将所有分量添加到这个标量变量中
loss = K.variable(0.)
layer_features = outputs_dict[content_layer]
target_image_features = layer_features[0, :, :, :]
combination_features = layer_features[2, :, :, :]
loss += content_weight * content_loss(target_image_features, combination_features)

# 添加每个目标层的风格损失分量到总的损失中
for layer_name in style_layers:
    layer_features = outputs_dict[layer_name]
    style_reference_features = layer_features[1, :, :, :]
    combination_features = layer_features[2, :, :, :]
    loss += (style_weight / len(style_layers)) * style_loss(style_reference_features, combination_features)
    pass
# 添加总变差损失
loss += total_variation_weight * total_variation_loss(combination_image)
# ----------------------------------------------------------------------
print("Listing 8.21 设置梯度下降过程")
# 获取损失相对于生成图像的梯度
grads = K.gradients(loss, combination_image)[0]
# 用于获取当前损失值和当前梯度值的函数
fetch_loss_and_grads = K.function([combination_image], [loss, grads])


# 这个类将 fetch_loss_and_grads 包装起来，因此可以利用两个单独的方法调用来获取损失和梯度
# 这两个函数就可以传入 SciPy 优化器
class Evaluator(object):
    def __init__(self):
        self.loss_value = None
        self.grads_values = None
        pass

    def loss(self, x):
        assert self.loss_value is None
        x = x.reshape((1, img_height, img_width, 3))
        outs = fetch_loss_and_grads([x])
        loss_value = outs[0]
        grad_values = outs[1].flatten().astype('float64')
        self.loss_value = loss_value
        self.grads_values = grad_values
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grads_values)
        self.loss_value = None
        self.grads_values = None
        return grad_values

    pass


evaluator = Evaluator()
# ----------------------------------------------------------------------
print("Listing 8.22 风格迁移循环")
from scipy.optimize import fmin_l_bfgs_b
from matplotlib.image import imsave
import time

result_prefix = 'my_result'
iterations = 20

# 目标图像是初始状态
x = preprocess_image(target_image_path)
# 将图像展平，因为 scipy.optimize。fmin_l_bfgs_b 只能处理展平的向量
x = x.flatten()
for i in range(iterations):
    print("\t迭代次数：第{}次".format(i))
    start_time = time.time()
    # fmin_l_bfgs_b(): Minimize a function func using the L-BFGS-B algorithm.
    # 对生成图像的像素运行 L-BFGS 最优化，将神经风格损失最小化。
    # 损失函数evaluator.loss()和梯度函数evaluator.grads()作为两个单独的参数传入
    x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x, fprime = evaluator.grads, maxfun = 20)
    print("\t当前的损失值:", min_val)

    # 将当前生成的图像保存
    img = x.copy().reshape((img_height, img_width, 3))
    img = deprocess_image(img)
    fname = result_prefix + '_at_iteration_{}.png'.format(i)
    imsave(fname, img)
    print("\t图像保存为：", fname)
    end_time = time.time()
    print("\tIteration {} completed n {}s".format(i, end_time - start_time))
# ----------------------------------------------------------------------
# 运行结束的提醒
winsound.Beep(600, 500)
if len(plt.get_fignums()) != 0:
    plt.show()
pass
