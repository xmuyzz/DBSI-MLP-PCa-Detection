import os
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import glob
from time import gmtime, strftime
from datetime import datetime
import timeit
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc, precision_recall_curve



def cm_all(proj_dir, output_dir, data_type):

    pro_data_dir = os.path.join(proj_dir, 'pro_data')

    if data_type == 'invivo':
        fn1 = 'invivo_voxel_pred.csv'
        fn2 = 'invivo_pat_pred.csv'
    elif data_type == 'exvivo':
        fn1 = 'exvivo_voxel_pred.csv'
        fn2 = 'exvivo_pat_pred.csv'

    # voxel level data
    df1 = pd.read_csv(os.path.join(pro_data_dir, fn1))
    y_test_vox = df1['y_test'].to_numpy()
    y_pred_vox = df1['y_pred'].to_numpy()
    y_pred1 = []
    for pred in y_pred_vox:
        if pred > 0.5:
            pred = 1
        else:
            pred = 0
        y_pred1.append(pred)
    y_pred1 = np.asarray(y_pred1)

    # patient level data prob
    df2 = pd.read_csv(os.path.join(pro_data_dir, fn2))
    y_test_pat = df2['y_test'].to_numpy()
    y_pred_pat1 = df2['y_pred'].to_numpy()
    y_pred2 = []
    for pred in y_pred_pat1:
        if pred > 0.5:
            pred = 1
        else:
            pred = 0
        y_pred2.append(pred)
    y_pred2 = np.asarray(y_pred2)

    # patient level data pos
    df2 = pd.read_csv(os.path.join(pro_data_dir, fn2))
    y_test_pat = df2['y_test'].to_numpy()
    y_pred_pat2 = df2['y_pred_class'].to_numpy()
    y_pred3 = []
    for pred in y_pred_pat2:
        if pred > 0.5:
            pred = 1
        else:
            pred = 0
        y_pred3.append(pred)
    y_pred3 = np.asarray(y_pred3)

    # ROC
    preds = [y_pred1, y_pred2, y_pred3]
    labels = [y_test_vox, y_test_pat, y_test_pat]
    level = ['voxel', 'patient_prob', 'patient_pos']

    for pred, label, level in zip(preds, labels, level):
        # using confusion matrix to calculate AUC
        cm = confusion_matrix(label, pred)
        cm_norm = cm.astype('float64') / cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.around(cm_norm, 2) 
        # classification report
        report = classification_report(label, pred, digits=3)
        print(level)
        print(cm)
        print(cm_norm)
        print(report)





