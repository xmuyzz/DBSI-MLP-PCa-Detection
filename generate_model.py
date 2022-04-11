import os
import timeit
import itertools
import numpy as np
import pandas as pd
import seaborn as sn
import glob2 as glob
from functools import partial
from datetime import datetime
import matplotlib.pyplot as plt
from time import gmtime, strftime

from tensorflow.keras import initializers
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Reshape, Activation, Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.layers import ELU, LeakyReLU
import tensorflow
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve, auc
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.utils import resample
from sklearn.metrics import roc_auc_score




def generate_model(init, dropout_rate, momentum, n_input, n_layer):

    model = Sequential()

    # input layer
    model.add(Dense(n_input, input_dim=n_input, kernel_initializer='he_uniform'))
    model.add(BatchNormalization(momentum=momentum))
    model.add(Activation('elu'))
    model.add(Dropout(dropout_rate))

    # hidden layer
    for i in range(n_layer):
        model.add(Dense(100))
        model.add(BatchNormalization(momentum=momentum))
        model.add(Activation('elu'))
        model.add(Dropout(dropout_rate))

    # output layer
    model.add(Dense(3))
    model.add(BatchNormalization(momentum=momentum))
    model.add(Activation('softmax'))


    return model



