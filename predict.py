import warnings
warnings.filterwarnings('ignore')
import os
import keras
print("Keras = {}".format(keras.__version__))
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
from matplotlib import cm
import numpy as np
import pylab
import sys
import math
from keras import backend as K
import keras.backend.tensorflow_backend as KTF
from keras.models import load_model
from sklearn import metrics
import pickle
import scipy.io as sio
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"]="1"

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.per_process_gpu_memory_fraction = 0.9
session = tf.Session(config = config)
KTF.set_session(session)

sys.setrecursionlimit(2000)


def bce_dice_loss(y_true, y_pred):
    return NMSE(y_true, y_pred)
    # return MAE(y_true, y_pred)
def NMSE(y_true, y_pred):
    a = K.sqrt(K.sum(K.square(y_true - y_pred)))
    b = K.sqrt(K.sum(K.square(y_true)))
    return a / b

def MAE(y_true, y_pred):
  a=K.mean(K.abs(y_true- y_pred))
  return a

HUBER_DELTA = 0.7
def smoothL1(y_true, y_pred):
   x   = K.abs(y_true - y_pred)
   x   = K.switch(x < HUBER_DELTA, 0.5 * x ** 2, HUBER_DELTA * (x - 0.5 * HUBER_DELTA))
   return  K.sum(x)

m=40
def mix(y_true, y_pred):
    a = K.sqrt(K.sum(K.square(y_true - y_pred)))
    b = K.sqrt(K.sum(K.square(y_true)))
    c = a / b
    d = K.mean(K.abs(y_true- y_pred))
    f = m*d + c
    return f

def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


indirect1=4096
batch_size=1

model_path = "./model/EDHR_Net_1/"

print('[*] load data ... ')
under=sio.loadmat('exp_ibuprofen_NUS.mat')
data=under['data']
X_test=data['input_x']
X_test=np.array(X_test[0,0],dtype=np.float32)
X_test=np.reshape(X_test, (batch_size,indirect1, 1, 1))

################################################

print('[*] define model ...')

model = load_model(os.path.join('./model/EDHR_Net_1', "EDHR_Net.h5"), custom_objects={'mix':mix,'NMSE':NMSE, 'smoothL1':smoothL1, 'tf':tf})
print('[*] start testing ...')

x_gen = model.predict([X_test], batch_size=batch_size, verbose=1)
x_gen=np.array(x_gen,dtype=np.float32)

sio.savemat('rec_spec_weak.mat',{'rec_spec':x_gen,'X_test':X_test})

print("[*] Job finished!")

