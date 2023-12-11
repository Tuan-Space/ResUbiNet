
import os

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import tensorflow as tf

# from getData import *
from DProcess import convertRawToXY
from capsule_keras import *
from assessment import *

from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
# from keras.layers import merge
import keras.layers.convolutional as conv
from tensorflow.keras import backend as K, regularizers
from keras.regularizers import l1, l2
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split, cross_val_score
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve, auc
from scipy import interpolate
from numpy import interp

# from models.attention import Attention, myFlatten

seq_len = 69
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.70)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

def shuffle_PosNeg(df):
    shuffled_df = df.sample(frac=1).reset_index(drop=True)
    return shuffled_df

# Load Data
train_data = pd.read_csv("../training_data_69.csv", sep=",")
test_data = pd.read_csv("../independent_data_69.csv", sep=",")
train_data = shuffle_PosNeg(train_data)
test_data = shuffle_PosNeg(test_data)
train_data = pd.DataFrame(train_data)
test_data = pd.DataFrame(test_data)

X_train, Y_train = convertRawToXY(np.array(train_data), codingMode=0)
X_test, Y_test = convertRawToXY(np.array(test_data), codingMode=0)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[2], X_train.shape[3])
X_test = X_test.reshape(X_test.shape[0], X_test.shape[2], X_test.shape[3])


def expand_dim_backend(x):
    x1 = K.reshape(x, (-1, 1, 256))
    return x1


def margin_loss(y_true, y_pred):
    lamb, margin = 0.5, 0.1
    return K.sum(y_true * K.square(K.relu(1 - margin - y_pred)) + lamb * (
            1 - y_true) * K.square(K.relu(y_pred - margin)), axis=-1)


def mul_model():
    optimizer = Adam(lr=0.0002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0001)
    input_one = Input(shape=(seq_len, 21))

    o_x = conv.Conv1D(filters=256, kernel_size=7,
                        activation='relu',
                        kernel_initializer='he_normal',
                        kernel_regularizer=regularizers.l2(0),
                        )(input_one)
    o_x = conv.Conv1D(filters=256, kernel_size=11,
                        activation='relu',
                        kernel_initializer='he_normal',
                        kernel_regularizer=regularizers.l2(0),
                        )(o_x)
    o_x = Dropout(0.3)(o_x)

    squeeze = GlobalAveragePooling1D()(o_x)
    squeeze = Lambda(expand_dim_backend)(squeeze)
    # squeeze = Lambda(expand_dim_backend)(o_x)
    excitation = Conv1D(filters=256, kernel_size=1, strides=1, padding='valid', activation='relu')(squeeze)
    excitation = Conv1D(filters=256, kernel_size=1, strides=1, padding='valid', activation='sigmoid')(excitation)
    o_x = Lambda(multiply)([o_x, excitation])

    out = Capsule(1, 10, 3, True)(o_x)

    output = Lambda(lambda x: K.sqrt(K.sum(K.square(x), 2)), output_shape=(1,))
    out = output(out)

    MDCaps = Model(input_one, out)

    MDCaps.compile(loss=margin_loss, optimizer=optimizer, metrics=["accuracy"])
    return MDCaps


n_split = 10
time = 0
tprs = []
aucs = []
fprs = []
all_pred_proba = []
mean_fpr = np.linspace(0, 1, 100)
true_label = list(Y_test)

for train_index, val_index in KFold(n_splits=n_split).split(X_train):
    X_one_train, X_one_val = X_train[train_index], X_train[val_index]
    Y_one_train, Y_one_val = Y_train[train_index], Y_train[val_index]

    models = mul_model()
    history = models.fit(X_one_train, Y_one_train, batch_size=128, epochs=45,
                         validation_data=(X_one_val, Y_one_val),
                         verbose=1)
    models.save_weights(f'./variant_result/model_{time+1}.h5')
    pred_proba = models.predict(X_test, batch_size=2048)
    pred_class = []
    for i in pred_proba:
        if i >= 0.5:
            i = 1
            pred_class.append(i)
        else:
            i = 0
            pred_class.append(i)
    all_pred_proba.append(pred_proba)

    acc, sensitivity, specificity, mcc, f1, precision = calculate_performance2(len(X_test), true_label, pred_class)
    time += 1
    print(time)
    print('acc:', acc)
    print('sen:', sensitivity)
    print('spe:', specificity)
    print('mcc:', mcc)
    print('f1', f1)
    print('precision:', precision)

    # ###### AUC ######
    fpr, tpr, thresholds = roc_curve(true_label, pred_proba)
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    # fprs.append(fpr)
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    print('AUC:', roc_auc)
    with open("variant_result/5cross-conv1d_2+SENet+CapsNet.csv", "a+") as f:
        f.write('time:' + str(time) + ' acc:' + str(acc) + '  sen:' + str(sensitivity) + '  spe:' + str(
            specificity) + '  mcc:' + str(mcc) + '  f1:' + str(f1) + '  AUC:' + str(roc_auc) +
                '  precision:' + str(precision) + '\n')

all_pred_class = []
all_pred_proba = sum(all_pred_proba) / 10
for i in all_pred_proba:
    if i >= 0.5:
        i = 1
        all_pred_class.append(i)
    else:
        i = 0
        all_pred_class.append(i)
acc, sensitivity, specificity, mcc, f1, precision = calculate_performance2(len(X_test), true_label, all_pred_class)

fpr, tpr, thresholds = roc_curve(true_label, all_pred_proba)
tprs.append(interp(mean_fpr, fpr, tpr))
tprs[-1][0] = 0.0
# fprs.append(fpr)
roc_auc = auc(fpr, tpr)
aucs.append(roc_auc)
print('AUC:', roc_auc)
with open("variant_result/5cross-conv1d_2+SENet+CapsNet.csv", "a+") as f:
    f.write('average' + ' acc:' + str(acc) + '  sen:' + str(sensitivity) + '  spe:' + str(
        specificity) + '  mcc:' + str(mcc) + '  f1:' + str(f1) + '  AUC:' + str(roc_auc) +
            '  precision:' + str(precision) + '\n')
