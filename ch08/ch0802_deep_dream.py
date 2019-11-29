# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   deep-learning-with-python-notebooks
@File       :   ch0802_deep_dream.py
@Version    :   v0.1
@Time       :   2019-11-28 14:46
@License    :   (C)Copyright 2018-2019, zYx.Tom
@Reference  :   《Python 深度学习，Francois Chollet》, Sec0802，P236
@Desc       :   生成式深度学习，使用 Keras 生成 Deep Dream
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np  # pip install numpy<1.17，小于1.17就不会报错
import winsound
# 屏蔽警告：Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
from keras.preprocessing import image
from matplotlib.image import imsave

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
from keras.applications import inception_v3
from keras import backend as K

# 使用下面这个命令可以禁止所有与训练有关的操作
# 因为参数全部使用全连接层 Inception V3 预训练 ImageNet 的权重
# 这个文件加载模型有点慢，没有训练只有梯度计算，因此计算结果比较快
print("Listing 8.8 加载预训练的 Inception V3 模型")
K.set_learning_phase(0)
model = inception_v3.InceptionV3(weights = 'imagenet', include_top = False)
# model = Sequential()
# ----------------------------------------------------------------------
print("Listing 8.9 设置 Deep Dream 配置")
layer_contributions = {
        'mixed2': 0.2, 'mixed3': 3., 'mixed4': 2., 'mixed5': 1.5,
}
# ----------------------------------------------------------------------
print("Listing 8.10 定义需要的最大化损失")
layer_dict = dict([(layer.name, layer) for layer in model.layers])
loss = K.variable(0.)
for layer_name in layer_contributions:
    coeff = layer_contributions[layer_name]
    activation = layer_dict[layer_name].output

    scaling = K.prod(K.cast(K.shape(activation), 'float32'))
    loss += coeff * K.sum(K.square(activation[:, 2:-2, 2:-2, :])) / scaling
    pass
# ----------------------------------------------------------------------
print("Listing 8.11 梯度上升过程")
dream = model.input
grads = K.gradients(loss, dream)[0]
grads /= K.maximum(K.mean(K.abs(grads)), 1e-7)

outputs = [loss, grads]
fetch_loss_and_grads = K.function([dream], outputs)
model.summary()


def eval_loss_and_grads(x):
    outs = fetch_loss_and_grads([x])
    loss_value = outs[0]
    grad_value = outs[1]
    return loss_value, grad_value


def gradient_ascent(x, iterations, step, max_loss = None):
    for i in range(iterations):
        loss_value, grad_value = eval_loss_and_grads(x)
        if max_loss is not None and loss_value > max_loss:
            break
        print("...Loss value at", i, ':', loss_value)
        x += step * grad_value
        pass
    return x


# ----------------------------------------------------------------------
# Listing 8.13 辅助函数
def resize_img(img, size):
    from scipy.ndimage import zoom
    img = np.copy(img)
    factors = (1, float(size[0]) / img.shape[1], float(size[1]) / img.shape[2], 1)
    return zoom(img, factors, order = 1)


# 用于打开图像，改变图像大小以及将图像格式转换为 Inception V3 模型能够处理的张量
def preprocess_image(image_path):
    img = image.load_img(image_path)
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis = 0)
    img = inception_v3.preprocess_input(img)
    return img


# 将一个张量转换为有效的图像
def deprocess_image(x):
    if K.image_data_format() == 'channels_first':
        x = x.reshape((3, x.shape[2], x.shape[3]))
        x = x.transpose([1, 2, 0])
    else:
        # 对inception_v3.preprocess_input()所做的预处理进行反向操作
        x = x.reshape((x.shape[1], x.shape[2], 3))
        pass
    x /= 2.
    x += 0.5
    x *= 255.
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def save_img(img, fname):
    pil_img = deprocess_image(np.copy(img))
    imsave(fname, pil_img)


# ----------------------------------------------------------------------
print("Listing 8.12 在多个连续尺度上运行梯度上升")
# 改变下面的超参数可以得到新的效果
step = 0.1  # 梯度上升的步长
iterations = 20  # 在每个尺度上运行梯度上升的步数
num_octave = 3  # 运行梯度上升的尺度个数
octave_scale = 1.4  # 两个尺度之间的大小比例
max_loss = 10.  # 如果最大损失超过10，就中断梯度上升的过程，避免得到丑陋的伪影

# 需要处理的图片路径
base_image_path = 'C:/Users/Administrator/PycharmProjects/Data/Pictures/original_photo_deep_dream.jpg'
img = preprocess_image(base_image_path)  # 将图片加载到 Numpy 数组
original_shape = img.shape[1:3]
successive_shapes = [original_shape]
# 定义运行梯度上升的不同尺度在下一个由形状元组组成的列表中
for i in range(1, num_octave):
    shape = tuple([int(dim / (octave_scale ** i)) for dim in original_shape])
    successive_shapes.append(shape)
    pass
successive_shapes = successive_shapes[::-1]  # 将形状列表反转，变为升序
# 将原始图片绽放到最小的尺寸
original_img = np.copy(img)
shrunk_original_img = resize_img(img, successive_shapes[0])
for shape in successive_shapes:
    print("Processing image shape", shape)
    img = resize_img(img, shape)  # 调整结果图片的尺寸
    img = gradient_ascent(img, iterations, step, max_loss)  # 基于梯度上升来改变结果图片
    # 将原始图像的较小版本放大会导致像素化
    upscaled_shrunk_original_img = resize_img(shrunk_original_img, shape)
    # 在这个尺寸上计算原始图像的高质量版本
    same_size_original = resize_img(original_img, shape)
    # 二者的差别就是在放大过程中丢失的细节
    lost_detail = same_size_original - upscaled_shrunk_original_img
    # 将丢失的细节重新注入到结果图片中
    img += lost_detail
    shrunk_original_img = resize_img(original_img, shape)
    save_img(img, 'dream_at_scale_' + str(shape) + '.png')
    pass
save_img(img, 'final_dream.png')

# ----------------------------------------------------------------------
# 运行结束的提醒
winsound.Beep(600, 500)
if len(plt.get_fignums()) != 0:
    plt.show()
pass
