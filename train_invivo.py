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
from sklearn.model_selection import cross_val_score, cross_val_predict
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
import random, copy
from crossValCreateMask import crossValCreateMask, ensemblePred

def train_invivo(model, x_train, y_train, x_val, y_val, x_test, y_test,
                 trainIDs, valIDs, testIDs, df_val, df_test, proj_dir, 
                 batch_size, epoch, loss, optimizer, folds, exclude_patient,
                 exclude_list, df_excludeX, df_excludeY, df_positions, makeMask, mapType):

    pro_data_dir = os.path.join(proj_dir, 'pro_data')
    def buildModel():
        model.compile(
            loss=loss,
            optimizer=optimizer,
            metrics=['accuracy']
            )
        return model
    
    estimator = KerasClassifier(build_fn=buildModel, epochs=epoch, batch_size=batch_size, verbose=1)

    crossValX = np.concatenate((x_train, x_test))
    crossValY = np.concatenate((y_train, y_test))

    # print(type(trainIDs))
    # print(type(valIDs))

    dataPatientIDs = np.array(list(trainIDs)+list(valIDs))
    allDataIdxs = range(dataPatientIDs.size)

    patientIDs = np.unique(dataPatientIDs)
    print('number of patients:')
    print(len(patientIDs))
    trainValIdxs = np.array(list(range(len(patientIDs))))
    shuffledIdxs = copy.deepcopy(trainValIdxs)
    random.shuffle(shuffledIdxs)
    # print(trainValIdxs)
    # print(shuffledIdxs)
    allTrainValParts = [None] * folds
    trainIdxs  = [None] * folds

    for i in range(folds):
        valIDs = patientIDs[shuffledIdxs[trainValIdxs % folds == i]]
        allTrainValParts[i] = []
        trainIdxs[i] = []
        for j in range(len(dataPatientIDs)):
            if dataPatientIDs[j] in valIDs:
                allTrainValParts[i].append(j)
            else:
                trainIdxs[i].append(j)
    valIdxs = np.array(allTrainValParts)
    trainIdxs = np.array(trainIdxs)

    print('kFold shuffling and splitting complete!')

    for i in range(folds):
        print("foldnum " + str(i))
        print(len(trainIdxs[i]))
        print(len(valIdxs[i]))
    def getTTS():
        splits = [None] * folds
        for i in range(folds):
            splits[i] = [trainIdxs[i], valIdxs[i]]
        return splits

    # splits = getTTS()
    # print(np.array(splits).shape)

    scores = cross_val_score(estimator, crossValX, crossValY, cv=getTTS(), scoring='accuracy')
    print('cross fold validation scores:')
    print(scores)
    print("%.2f (%.2f) MSE" % (scores.mean(), scores.std()))

    model.save_weights(os.path.join(pro_data_dir, 'invivoModelInitialWts.h5'))
    #^^ensures every training model has "clean slate" of weights so it is not incrementally trained
    modelNames = [None] * folds
    histories = [None] * folds
    for i in range(folds):
        model.load_weights(os.path.join(pro_data_dir, 'invivoModelInitialWts.h5'))
        histories[i] = model.fit(
            x=crossValX[trainIdxs[i]],
            y=crossValY[trainIdxs[i]],
            batch_size=batch_size,
            epochs=epoch,
            verbose=1,
            callbacks=None,
            validation_split=None,
            validation_data=(crossValX[valIdxs[i]], crossValY[valIdxs[i]]),
            shuffle=True,
            class_weight=None,
            sample_weight=None,
            initial_epoch=0,
            steps_per_epoch=None,
            validation_steps=None,
            )
        saveName = "valModel"+ str(i+1) +".h5"
        modelNames[i] = saveName
        model.save(os.path.join(pro_data_dir,saveName))
    #TODO: make matrix of image size filled with 0, make predictions on whatever the data stuff is,
    #get the coordinates and fill in matrix from there. Save models. In seperate code, load models,
    #Take average from all 5 cross-val models, do a majority vote, then final matrix turns into mask.
    print("cross validation models saved! Creating masks on excluded data...")
    if exclude_patient:
        crossValCreateMask(modelNames, proj_dir, df_excludeX, df_excludeY, df_positions, exclude_list, makeMask, mapType)

    print("Cross Validation masks done! Developing overall model:")
    model.load_weights(os.path.join(pro_data_dir, 'invivoModelInitialWts.h5'))
    history = model.fit(
        x=x_train,
        y=y_train,
        batch_size=batch_size,
        epochs=epoch,
        verbose=1,
        callbacks=None,
        validation_split=None,
        validation_data=(x_val, y_val),
        shuffle=True,
        class_weight=None,
        sample_weight=None,
        initial_epoch=0,
        steps_per_epoch=None,
        validation_steps=None,
        )
    y_pred = model.predict(x_test)
    y_pred_class = np.argmax(y_pred, axis=1)
    score = model.evaluate(x_test, y_test, verbose=0)
    loss = np.around(score[0], 3)
    acc = np.around(score[1], 3)
    print('acc:', acc)
    print('loss:', loss)

    # save model
    #-------------
    model.save(os.path.join(pro_data_dir, 'invivo_model.h5'))

    # save a df for test and prediction
    #------------------------------------
    df_test['y_pred'] = y_pred[:, 1]
    df_test['y_pred_class'] = y_pred_class
    df_test.rename(columns={'ROI_Class': 'y_test'}, inplace=True)
    test_voxel_pred = df_test[['Sub_ID', 'y_test', 'y_pred', 'y_pred_class']]
    test_voxel_pred.to_csv(os.path.join(pro_data_dir, 'invivo_voxel_pred.csv'))
    print('successfully save test voxel prediction!')

    # get pred class on patient level
    #-------------------------------------
    df_mean = test_voxel_pred.groupby(['Sub_ID'], as_index=False).mean()
    #print(df_mean)
    label_pat = df_mean['y_test'].to_numpy()
    pred_pat = df_mean['y_pred'].to_numpy()
    #print(label_pat)
    #print(pred_pat)
    
    pred_class_pat = []
    for pred in pred_pat:
        if pred > 0.5:
            pred = 1
        else:
            pred = 0
        pred_class_pat.append(pred)
    pred_class_pat = np.asarray(pred_class_pat)
    df_mean['y_pred_class'] = pred_class_pat
    df_mean['y_test'] = label_pat
    df_mean['y_pred'] = pred_pat
    test_pat_pred = df_mean[['Sub_ID', 'y_test', 'y_pred', 'y_pred_class']]
    print(df_mean)
    test_pat_pred.to_csv(os.path.join(pro_data_dir, 'invivo_pat_pred.csv'))
    print('successfully save test patient prediction!')
