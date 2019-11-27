# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   deep-learning-with-python-notebooks
@File       :   ch0504_visualizing_heatmaps_activation.py
@Version    :   v0.1
@Time       :   2019-11-22 16:32
@License    :   (C)Copyright 2018-2019, zYx.Tom
@Reference  :   《Python 深度学习，Francois Chollet》, Sec05，P
@Desc       :   深度学习用于计算机视觉，卷积神经网络简介
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np  # pip install numpy<1.17，小于1.17就不会报错
import winsound
from keras import backend as K
from keras.applications import VGG16
from keras.applications.vgg16 import decode_predictions, preprocess_input
from keras.preprocessing import image

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

# 图片下载：https://s3.amazonaws.com/book.keras.io/img/ch5/creative_commons_elephant.jpg
model = VGG16(weights = 'imagenet')
img_path = "C:/Users/Administrator/PycharmProjects/Data/Pictures/"
img_name = "creative_commons_elephant.jpg"
img_cam_name = "elephant_cam.jpg"
img = image.load_img(img_path + img_name, target_size = (224, 224))
x = image.img_to_array(img)  # 形状为（224，224，3）的float32格式的Numpy数组
x = np.expand_dims(x, axis = 0)  # 形状为（1，224，224，3）的float32格式的Numpy数组
x = preprocess_input(x)  # 对输入数据进行预处理（按通道进行颜色标准化）
# 使用VGG16模型预测图片的类别
preds = model.predict(x)
print("预测结果：", decode_predictions(preds, top = 3)[0])
print("预测结果对应的编号：", np.argmax(preds[0]))

print("Code 5-42：使用 Grid-CAM 算法寻找图像中哪些部分最像非洲象")
african_elephant_output = model.output[:, 386]
last_conv_layer = model.get_layer('block5_conv3')
# K.gradients()报错cost must be a scalar。是因为 Theano 不支持，请切换成 TensorFlow
grads = K.gradients(african_elephant_output, last_conv_layer.output)[0]
pooled_grads = K.mean(grads, axis = (0, 1, 2))
iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
pooled_grads_value, conv_layer_output_value = iterate([x])
for i in range(512):
    conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
    pass

print("Code 5-43：绘制热力图")
heatmap = np.mean(conv_layer_output_value, axis = -1)
heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)
plt.matshow(heatmap)
plt.title("图5-35：测试图像的“非洲象”类激活热力图")

print("Code 5-44：将热力图与原始图像叠加")
import cv2

# 使用 CV2 加载原始图像
img = cv2.imread(img_path + img_name)
# 将热力图的大小跟图像大小调整为一样
heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
# 将热力图转换为RGB格式
heatmap = np.uint8(255 * heatmap)
# ToSee：将热力图转换为颜色图？
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
# 热力图强度叠加
superimposed_img = heatmap * 0.4 + img
# matplotlib 显示的图片可能是色彩深度不够，效果不太漂亮，不满意可以打开自己保存的图
cv2.imwrite(img_path + img_cam_name, superimposed_img)
plt.figure()
plt.imshow(superimposed_img.astype('uint8'))
plt.title("图5-36：将类激活热力图叠加到原始图像上")

# 运行结束的提醒
winsound.Beep(600, 500)
if len(plt.get_fignums()) != 0:
    plt.show()
pass
