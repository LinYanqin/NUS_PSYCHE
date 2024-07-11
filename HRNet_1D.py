from __future__ import print_function
import keras
import tensorflow as tf
from keras import backend as K
from keras.layers import merge, Input,UpSampling2D, Conv2D,concatenate,ZeroPadding2D, Conv2DTranspose,\
    AveragePooling2D,add, SeparableConv2D
from    tensorflow.keras import layers, Sequential, Model
from    tensorflow.keras.layers import Conv1D, BatchNormalization, Activation, add, LeakyReLU
from keras.layers.pooling import MaxPooling2D, GlobalAveragePooling2D
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import BatchNormalization, Dropout, Flatten, Lambda
from keras.layers.advanced_activations import ELU, LeakyReLU
from keras.optimizers import Adam, RMSprop, SGD
from keras.regularizers import l2
from keras.layers.noise import GaussianDropout
from keras.models import Model
import numpy as np
from model_logic2 import *

g_init = tf.random_normal_initializer(1.,0.02)    ##生成标准正态分布的随机数

from keras import layers
from keras.applications.imagenet_utils import (decode_predictions,
                                               preprocess_input)
from keras.preprocessing import image
from keras.utils.data_utils import get_file
s=25

def conv2_x(input_tensor, filters, stage, block, w_init):
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    con = SeparableConv2D(filters, (s, 1), strides=(1, 1), activation=None,
                            name=conv_name_base + '2a', kernel_initializer=w_init,
                            bias_initializer='zeros', padding='same')(input_tensor)
    bn = BatchNormalization(gamma_initializer=g_init,name=bn_name_base + '2a')(con)
    ac = LeakyReLU(alpha=0.2)(bn)

    con = SeparableConv2D(filters, (s, 1), strides=(1, 1), activation=None,
                            name=conv_name_base + '2b', kernel_initializer=w_init,
                            bias_initializer='zeros', padding='same')(ac)
    bn = BatchNormalization(gamma_initializer=g_init,name=bn_name_base + '2b')(con)
    conv2_x_add = layers.add([bn, input_tensor])

    ac = LeakyReLU(alpha=0.2)(conv2_x_add)
    ac = Dropout(0.4)(ac)
    return ac



def ResNet50_2(img_rows, img_cols, color_type=1, num_class=1):
    w_init = 'glorot_normal'
    filters = 32

    img_input = Input(shape=(color_type, img_rows, img_cols), name='main_input')
    img_input_2 = Input(shape=(color_type, img_rows, img_cols), name='main_input2')
    combined=concatenate([img_input,img_input_2])

    x = SeparableConv2D(filters, (15, 1), strides=(1, 1), activation=None, kernel_initializer=w_init,
                                  bias_initializer='zeros', padding='same')(combined)
    x = BatchNormalization(name='bn_conv2')(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = conv2_x(x, filters, 2, 'b2', w_init)
    x = conv2_x(x, filters, 2, 'c2', w_init)
    x = conv2_x(x, filters, 2, 'd2', w_init)
    x = conv2_x(x, filters, 2, 'f2', w_init)
    x = conv2_x(x, filters, 2, 'g2', w_init)
    x = conv2_x(x, filters, 2, 'h2', w_init)
    x = conv2_x(x, filters, 2, 'i2', w_init)
    x = conv2_x(x, filters, 2, 'k2', w_init)

    x = SeparableConv2D(num_class, (1, 1), strides=(1, 1), activation='relu', name='output',
                        kernel_initializer = w_init, bias_initializer='zeros', padding='same',
                        kernel_regularizer=l2(1e-3))(x)

    model = Model(input=[img_input,img_input_2], output=x)
    return model
