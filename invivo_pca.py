
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


# ----------------------------------------------------------------------------------
# construct train, validation and test dataset
# ----------------------------------------------------------------------------------
def get_data(proj_dir, benign_bix, benign_nobix, pca_bix, exclude_patient, 
             exclude_list, x_input):

    data_dir = os.path.join(proj_dir, 'data')
    
    maps_list = [
         'b0_map.nii',                       #07
         'dti_adc_map.nii',                  #08
         'dti_axial_map.nii',                #09
         'dti_fa_map.nii',                   #10
         'dti_radial_map.nii',               #11
         'fiber_ratio_map.nii',              #12
         'fiber1_axial_map.nii',             #13
         'fiber1_fa_map.nii',                #14
         'fiber1_fiber_ratio_map.nii',       #15
         'fiber1_radial_map.nii',            #16
         'fiber2_axial_map.nii',             #17
         'fiber2_fa_map.nii',                #18
         'fiber2_fiber_ratio_map.nii',       #19
         'fiber2_radial_map.nii',            #20
         'hindered_ratio_map.nii',           #21
         'hindered_adc_map.nii',             #22
         'iso_adc_map.nii',                  #23
         'restricted_adc_1_map.nii',         #24
         'restricted_adc_2_map.nii',         #25
         'restricted_ratio_1_map.nii',       #26
         'restricted_ratio_2_map.nii',       #27
         'water_adc_map.nii',                #28
         'water_ratio_map.nii',              #29
        ]
    
    dfs = []
    for data in [benign_nobix, benign_bix, pca_bix]:
        df = pd.read_csv(os.path.join(data_dir, data))
        df['ROI_Class'].replace(['p', 'c', 't'], [0, 0, 1], inplace=True)
        df.fillna(0, inplace=True)
        dfs.append(df)
    df1 = dfs[0]
    df2 = dfs[1]
    df3 = dfs[2]

    ## exlude 5 patients for validation
    if exclude_patient == True:
        print('exclude pat list:', exclude_list)
        indx = df3[df3['Sub_ID'].isin(exclude_list)].index
        print('total voxel:', df3.shape[0])
        df3.drop(indx, inplace=True)
        print('total voxel:', df3.shape[0])
    else:
        df3 = df3

    ## split PCa cohorts to train and test
    train_inds, test_inds = next(GroupShuffleSplit(
        test_size=0.5,
        n_splits=2, 
        random_state=7).split(df3, groups=df3['Sub_ID']))
    df3_train = df3.iloc[train_inds]
    df3_test = df3.iloc[test_inds]
    
    ## create train and test sets
    df_trainval = pd.concat([df1, df3_train])
    df_test = pd.concat([df2, df3_test])

    ## split PCa cohorts to train and test
    train_inds, val_inds = next(GroupShuffleSplit(
        test_size=0.2,
        n_splits=2,
        random_state=7).split(df_trainval, groups=df_trainval['Sub_ID']))
    df_train = df_trainval.iloc[train_inds]
    df_val = df_trainval.iloc[val_inds]

    # get data and labels
    x_train = df_train.iloc[:, x_input]
    y_train = df_train['ROI_Class'].astype('int')
    x_val = df_val.iloc[:, x_input]
    y_val = df_val['ROI_Class'].astype('int')
    x_test = df_test.iloc[:, x_input]
    y_test = df_test['ROI_Class'].astype('int')
   
    # scale data#
    x_train = MinMaxScaler().fit_transform(x_train)
    x_val = MinMaxScaler().fit_transform(x_val)
    x_test = MinMaxScaler().fit_transform(x_test)

    pd.set_option('display.max_columns', 500)
    pd.set_option('display.max_rows', 500)
    #print(x_val[0:100])
    #print(y_val[4500:5000])
    print('train size:', len(y_train))
    print('val size:', len(y_val))
    print('test size:', len(y_test))

    return x_train, y_train, x_val, y_val, x_test, y_test, df_val, df_test

# ----------------------------------------------------------------------------------
# construct DNN model with batch normalization layers and dropout layers
# ----------------------------------------------------------------------------------
def dnn_model(init, optimizer, loss, dropout_rate, momentum, n_input, n_layer):

    model = Sequential()

    # input layer    
    model.add(Dense(n_input, input_dim=n_input, kernel_initializer='he_uniform'))
    model.add(BatchNormalization(momentum=momentum))
    model.add(Activation('elu'))
    model.add(Dropout(dropout_rate))
    
    # hidden layer
    for i in range(10):
        model.add(Dense(100))
        model.add(BatchNormalization(momentum=momentum))
        model.add(Activation('elu'))
        model.add(Dropout(dropout_rate))
    
    # output layer
    model.add(Dense(3))
    model.add(BatchNormalization(momentum=momentum))
    model.add(Activation('softmax'))

    #model.summary()
    model.compile(
        loss=loss,
        optimizer=optimizer,
        metrics=['accuracy']
        )
    
    return model

# ----------------------------------------------------------------------------------
# trainning DNN model
# ----------------------------------------------------------------------------------
def model_train(model, x_train, y_train, x_val, y_val, x_test, y_test, 
                df_val, df_test, proj_dir):
    
    pro_data_dir = os.path.join(proj_dir, 'pro_data')

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
    
    ## save model
    model.save(os.path.join(pro_data_dir, 'saved_model.h5')) 
    
    ### save a df for test and prediction
    df_test['y_pred'] = y_pred[:, 1]
    df_test['y_pred_class'] = y_pred_class
    test_voxel_pred = df_test[['Sub_ID', 'ROI_Class', 'y_pred', 'y_pred_class']]
    test_voxel_pred.to_csv(os.path.join(pro_data_dir, 'test_voxel_pred.csv'))
    
    ## patient level
    df_mean = test_voxel_pred.groupby(['Sub_ID'], as_index=False).mean()
    #print(df_mean)
    label_pat = df_mean['ROI_Class'].to_numpy()
    pred_pat = df_mean['y_pred'].to_numpy()
    #print(label_pat)
    #print(pred_pat)
    # get pred class on patient level
    pred_class_pat = []
    for pred in pred_pat:
        if pred > 0.5:
            pred = 1
        else:
            pred = 0
        pred_class_pat.append(pred)
    pred_class_pat = np.asarray(pred_class_pat)
    df_mean['pred_class_pat'] = pred_class_pat
    df_mean['label_pat'] = label_pat
    df_mean['pred_pat'] = pred_pat
    test_pat_pred = df_mean[['Sub_ID', 'label_pat', 'pred_pat', 'pred_class_pat']]
    print(df_mean)
    test_pat_pred.to_csv(os.path.join(pro_data_dir, 'test_pat_pred.csv'))
    
    # get confusiom matrix
    y_test = np.asarray(y_test)
    preds = [y_pred_class, pred_class_pat]
    labels = [y_test, label_pat]
    level = ['voxel', 'patient']
    for pred, label, level in zip(preds, labels, level):
        cm = confusion_matrix(pred, label)
        cm_norm = cm.astype('float')/cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.around(cm_norm, 3)
        report = classification_report(pred, label, digits=3)
        print('prediction level:', level)
        print('confusion matrix:')
        print(cm)
        print(cm_norm)
        print(report)

    ## ROC
    pred_voxel = y_pred[:, 1]
    preds = [pred_voxel, pred_pat]
    labels = [y_test, label_pat]
    level = ['voxel', 'patient']
    for pred, label, level in zip(preds, labels, level):
        fpr, tpr, threshold = roc_curve(label, pred)
        index = range(len(pred))
        #print(len(index))
        indices = resample(index, replace=True, n_samples=int(len(pred)))
        #print(len(indices))
        fpr, tpr, thre = roc_curve(label[indices], pred[indices])
        q = np.arange(len(tpr))
        roc = pd.DataFrame(
            {'fpr' : pd.Series(fpr, index=q),
             'tpr' : pd.Series(tpr, index=q),
             'tnr' : pd.Series(1 - fpr, index=q),
             'tf'  : pd.Series(tpr - (1 - fpr), index=q),
             'thre': pd.Series(thre, index=q)}
             )
        ### calculate optimal TPR, TNR under uden index
        roc_opt = roc.loc[(roc['tpr'] - roc['fpr']).idxmax(), :]
        AUC = roc_auc_score(label[indices], pred[indices])
        TPR = roc_opt['tpr']
        TNR = roc_opt['tnr']
        THR = roc_opt['thre']
        print('prediction level:', level)
        print('ROC AUC:', np.around(AUC, 3))
        print('ROC TPR:', np.around(TPR, 3))
        print('ROC TNR:', np.around(TNR, 3))
        print('ROC THR:', np.around(THR, 3))
    
    return loss, acc, test_pat_pred


# ----------------------------------------------------------------------------------
# plot confusion matrix
# ----------------------------------------------------------------------------------
def plot_CM(CM, fmt):
    
    ax = sn.heatmap(
        CM,
        annot=True,
        cbar=True,
        cbar_kws={'ticks': [-0.1]},
        annot_kws={'size': 22, 'fontweight': 'bold'},
        cmap="Blues",
        fmt=fmt,
        linewidths=0.5
        )
    ax.axhline(y=0, color='k', linewidth=4)
    ax.axhline(y=3, color='k', linewidth=4)
    ax.axvline(x=0, color='k', linewidth=4)
    ax.axvline(x=3, color='k', linewidth=4)
    ax.tick_params(direction='out', length=4, width=2, colors='k')
    ax.xaxis.set_ticks_position('top')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_aspect('equal')
    plt.tight_layout()
    cm_fn = 'cm.png' 
    plt.savefig(
        os.path.join(output_dir, cm_fn),
        format='png',
        dpi=600
        )
    plt.close()
    

# ----------------------------------------------------------------------------------
# run the model
# ---------------------------------------------------------------------------------- 
if __name__ == '__main__':

    proj_dir = '/home/xmuyzz/Harvard_AIM/others/pca'
    output_dir  = '/mnt/aertslab/USERS/Zezhong/others/pca/output'
    
    benign_nobix = 'benign_no_biopsy.csv'
    benign_bix = 'benign_biopsy.csv'
    pca_bix = 'pca_biopsy.csv'
    exclude_patient = False
    exclude_list = ['001_ZHOU_CHAO_GANG', '002_ZHU_XIN_GEN', '007_SHEN_QIU_YU',
                    '016_LIU_FENG_MEI', '028_XUE_LUO_PING']
    alpha = 0.3
    random_state = 42
    ELU_alpha = 1.0
    digit = 3
    n_layer = 5
    #x_input = [12, 13, 14, 15, 16, 21, 22, 23, 24, 25, 26, 28, 29]
    x_input = range(7, 30)
    #x_input = [7, 8]
    n_input = len(x_input)
    n_output = 3
    count = 0
    lr = 0.001
    momentum = 0.97
    dropout_rate = 0.3
    batch_size = 256
    epoch = 1
    n_neurons = 100    
    init = 'he_uniform' 
    optimizer = Adam(lr=lr)
    loss = 'sparse_categorical_crossentropy'
    output_activation = 'softmax'
    activation = ELU(alpha=ELU_alpha)     
    
    '''
    keranl initializer: 'he_uniform', 'lecun_normal', 'lecun_uniform'
    optimizer function: 'adam', 'adamax', 'nadam', 'sgd'
    loss function: 'categorical_crossentropy'
    activation function: LeakyReLU(alpha=alpha)
    '''

    # ----------------------------------------------------------------------------------
    # run the model
    # ----------------------------------------------------------------------------------
    start = timeit.default_timer()
    
    x_train, y_train, x_val, y_val, x_test, y_test, df_val, df_test = get_data(
        proj_dir=proj_dir, 
        benign_bix=benign_bix, 
        benign_nobix=benign_nobix, 
        pca_bix=pca_bix,
        exclude_patient=exclude_patient,
        exclude_list=exclude_list, 
        x_input=x_input
        )
    
    model = dnn_model(
        init=init,
        optimizer=optimizer,
        loss=loss,
        dropout_rate=dropout_rate,
        momentum=momentum,
        n_input=n_input,
        n_layer=n_layer
        )
    
    loss, acc, df_test_pred = model_train(
        model=model,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
        df_val=df_val,
        df_test=df_test,
        proj_dir=proj_dir
        )
    
    #plot_CM(cm_norm, '')

    # ----------------------------------------------------------------------------------
    # confusion matrix, sensitivity, specificity, presicion, f-score, model parameters
    # ----------------------------------------------------------------------------------
    print('epochs:        ', epoch)
    print('batch size:    ', batch_size)
    print('dropout rate:  ', dropout_rate)
    print('batch momentum:', momentum)
    print('learning rate: ', lr)
    print("train dataset size:", len(x_train))
    print("validation dataset size:", len(x_val))
    print("test dataset size:", len(x_test))       
    stop = timeit.default_timer()
    running_seconds = np.around(stop - start, 0)
    running_minutes = np.around(running_seconds/60, 0)
    print('\nDNN Running Time:', running_minutes, 'minutes')



