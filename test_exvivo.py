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
from imblearn.over_sampling import SMOTE, SVMSMOTE, KMeansSMOTE
from tensorflow.keras import initializers
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.models import Sequential, Model, load_model
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



def test_exvivo(proj_dir, output_dir, wu_split, x_test, y_test, df_test, x_test1, y_test1, 
                x_test2, y_test2, df_test1, df_test2):
    
    if wu_split == True:
        x_test = x_test1
        y_test = y_test1
        df_test = df_test1
    else:
        x_test = x_test
        y_test = y_test
        df_test = df_test

    pro_data_dir = os.path.join(proj_dir, 'pro_data')
    tuned_model = load_model(os.path.join(pro_data_dir, 'Tuned_model'))
    y_pred = tuned_model.predict(x_test)  
    y_pred_class = np.argmax(y_pred, axis=1)
    score = tuned_model.evaluate(x_test, y_test, verbose=0)
    loss = np.around(score[0], 3)
    acc = np.around(score[1], 3)
    print('acc:', acc)
    print('loss:', loss)
    
    # save a df for test on voxel
    df_test['y_pred'] = y_pred[:, 1]
    df_test['y_pred_class'] = y_pred_class
    df_test.rename(columns={'ROI_Class': 'y_test'}, inplace=True)
    test_voxel_pred = df_test[['ID', 'y_test', 'y_pred', 'y_pred_class']]
    test_voxel_pred.to_csv(os.path.join(pro_data_dir, 'exvivo_voxel_pred.csv'))
    print('successfully save test voxel prediction!')

    # get pred class on patient level
    df_mean = test_voxel_pred.groupby(['ID'], as_index=False).mean()
    print(df_mean)
    label_pat = df_mean['y_test'].to_numpy()
    pred_pat = df_mean['y_pred'].to_numpy()
    print(label_pat)
    print(pred_pat)
    
    pred_class_pat = []
    for pred in pred_pat:
        if pred > 0.16:
            pred = 1
        else:
            pred = 0
        pred_class_pat.append(pred)
    pred_class_pat = np.asarray(pred_class_pat)
    df_mean['pred_class_pat'] = pred_class_pat
    df_mean['y_test'] = label_pat
    df_mean['y_pred'] = pred_pat
    test_pat_pred = df_mean[['ID', 'y_test', 'y_pred', 'y_pred_class']]
    print(test_pat_pred)
    print('patient label:', test_pat_pred.groupby('y_test').count())
    test_pat_pred.to_csv(os.path.join(pro_data_dir, 'exvivo_pat_pred.csv'))
    print('successfully save test patient prediction!')



