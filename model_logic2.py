'''

'''
import keras
import tensorflow as tf
from keras.models import Model
from keras import backend as K
from keras.layers import Input, merge, Conv2D, ZeroPadding2D, UpSampling2D, Dense, concatenate, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D, GlobalAveragePooling2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import BatchNormalization, Dropout, Flatten, Lambda
from keras.layers.advanced_activations import ELU, LeakyReLU
from keras.optimizers import Adam, RMSprop, SGD
from keras.regularizers import l2
from keras.layers.noise import GaussianDropout
from keras.applications import VGG16
import numpy as np

smooth = 1.
dropout_rate = 0.5
act = "relu"

# Custom loss function
def FID_loss(y_true, y_pred):
    complex_true = tf.complex(y_true[:,:,0], y_true[:,:,1])
    complex_pred = tf.complex(y_pred[:,:,0], y_pred[:,:,1])
    ifft_true = tf.spectral.ifft2d(complex_true)
    ifft_pred = tf.spectral.ifft2d(complex_pred)
    y_true = tf.stack([tf.real(ifft_true),tf.imag(ifft_true)],axis=2)
    y_pred = tf.stack([tf.real(ifft_pred),tf.imag(ifft_pred)],axis=2)
    return NMSE(y_true, y_pred)

def NMSE(y_true, y_pred):
    a = K.sqrt(K.sum(K.square(y_true - y_pred)))
    b = K.sqrt(K.sum(K.square(y_true)))
    return a / b

HUBER_DELTA = 1./9
def smoothL1(y_true, y_pred):
   x   = K.abs(y_true - y_pred)
   x   = K.switch(x < HUBER_DELTA, 0.5 * x ** 2, HUBER_DELTA * (x - 0.5 * HUBER_DELTA))
   return  K.sum(x)

def msle(y_true, y_pred):
    first_log = K.log(K.clip(y_pred, K.epsilon(), None) + 1.)
    second_log = K.log(K.clip(y_true, K.epsilon(), None) + 1.)
    return K.mean(K.square(first_log - second_log), axis=-1)

m=3000
def mix(y_true, y_pred):
    a = K.sqrt(K.sum(K.square(y_true - y_pred)))
    b = K.sqrt(K.sum(K.square(y_true)))
    c = a / b
    d = K.mean(K.abs(y_true- y_pred))
    f = m*d + c
    return f

def MSE(y_true, y_pred):
    return keras.losses.mean_squared_error(y_true, y_pred) 

def MAE(y_true, y_pred):
  a=K.mean(K.abs(y_true- y_pred))
  return a

def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)

def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)



