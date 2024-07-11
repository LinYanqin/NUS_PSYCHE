import numpy as np
import os
import tensorflow as tf
import time
from scipy.io import loadmat

indirect=4096     #point of data

def data_generator(f_xtrain, f_ytrain, batch_size):
    idx = np.arange(len(f_xtrain))
    np.random.shuffle(idx)
    batches = [idx[range(batch_size*i, min(len(f_xtrain), batch_size*(i+1)))] for i in range(len(f_xtrain)//batch_size)]
    while True:
        for batch in batches:
            x_train = []
            y_train = []
            Xtrain = []
            Ytrain = []
            X = np.zeros((len(batch),indirect, 1, 1))
            Y = np.zeros((len(batch),indirect, 1, 1))
            for i in batch:
                Xtrain.append(f_xtrain[i])
                Ytrain.append(f_ytrain[i])
            for i in range(len(batch)):
                x_train_temp = []
                y_train_temp = []
                file_path = Xtrain[i]
                file_path2 = Ytrain[i]
                fr = open(file_path)
                for line in fr.readlines():
                    lineArr = line.strip().split()
                    x_train_temp.append(lineArr[:])
                x_train_temp = np.asarray(x_train_temp, dtype=np.float32)
                x_train_temp = np.reshape(x_train_temp, (indirect, 1, 1), 'F')
                lineArr=[]
                fr2 = open(file_path2)
                for line in fr2.readlines():
                    lineArr = line.strip().split()
                    y_train_temp.append(lineArr[:])
                lineArr=[]
                y_train_temp = np.asarray(y_train_temp, dtype=np.float32)
                y_train_temp = np.reshape(y_train_temp, (indirect, 1, 1), 'F')
                X[i,:,:,:]=x_train_temp
                Y[i,:,:,:]=y_train_temp
            yield(X, Y)
