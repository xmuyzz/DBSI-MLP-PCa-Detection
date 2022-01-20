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
from tensorflow import keras

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
import nibabel as nib

def ensemblePred(models, pro_data_dir, xData, modelType = 'class', positions = None, voteThresh = 0.5, mask = None):
    #type can be 'class' or 'raw'; raw gives total raw votes, class does voting
    allPred = np.zeros(xData.shape[0])
    ensemblePred = np.zeros(xData.shape[0])
    for modelName in models:
        model = keras.models.load_model(os.path.join(pro_data_dir, modelName))
        modelPred = model.predict(xData)
        modelPredClass = np.argmax(modelPred, axis=1)
        allPred = allPred + modelPredClass
    
    if modelType == 'class':
        ensemblePred = np.array(allPred) > voteThresh*(len(models))
    else:
        ensemblePred = allPred

    if mask is not None and positions is not None:
        maskDim = mask
        maskOutput = np.zeros(maskDim)
        positions.columns = positions.columns.str.replace(' ', '')
        # print(positions.columns)
        for i in range(len(ensemblePred)):
            [X, Y, Z] = positions.iloc[i,1:4]
            # X = positions.loc['X'][i]
            # Y = positions.loc['Y'][i]
            # Z = positions.loc['Z'][i]
            maskOutput[X,Y,Z] = ensemblePred[i]
        print(maskOutput.shape)
        return maskOutput
    else:
        return ensemblePred

def crossValCreateMask(models, proj_dir, df_excludeX, df_excludeY, df_positions, exclude_list, makeMask, mapType):
    pro_data_dir = os.path.join(proj_dir, 'pro_data')
    masks_dir = os.path.join(proj_dir, 'masks')
    outputs = []
    maskDim = [280, 224, 24]
    indexCounter = 0
    for excluded in exclude_list:
        indicesToPredict = df_positions['Sub_ID'] == excluded
        # print(indicesToPredict.values)
        xData = df_excludeX.iloc[indicesToPredict.values,:]
        if makeMask:
            maskArray = ensemblePred(models, pro_data_dir, xData, modelType = 'class', positions = df_positions.iloc[indicesToPredict.values,:], voteThresh = 0.5, mask = maskDim)
            outputs.append(maskArray)
            affineFile = nib.load(os.path.join(masks_dir, excluded, mapType))
            predMask = nib.Nifti1Image(np.array(maskArray).astype(int), affineFile.affine)
            nib.save(predMask,os.path.join(masks_dir, excluded,'tumorPred.nii'))
        else:
            outputs.append([ensemblePred(models, pro_data_dir, xData, modelType = 'class', voteThresh = 0.5)])

        indexCounter = indexCounter + 1
    # print(outputs[0])

#----------------------------------------TESTING CODE BELOW----------------------------------------
models = ['valModel1.h5','valModel2.h5','valModel3.h5']
proj_dir = r'C:\Users\atwu\Desktop\PCa_voxel_data'
pro_data_dir = os.path.join(proj_dir,'pro_data')
xData = pd.read_csv(os.path.join(pro_data_dir, 'excludeX.csv'), index_col=0)
yData = pd.read_csv(os.path.join(pro_data_dir, 'excludeY.csv'), index_col=0)
mapType = 'dti_adc_map.nii'
excludeList =  ['001_ZHOU_CHAO_GANG', '002_ZHU_XIN_GEN', '007_SHEN_QIU_YU',
                    '016_LIU_FENG_MEI', '028_XUE_LUO_PING']
positions = pd.read_csv(os.path.join(pro_data_dir, 'excludePositions.csv'), index_col=0)

crossValCreateMask(models, proj_dir, xData, yData, positions, excludeList, True, mapType)