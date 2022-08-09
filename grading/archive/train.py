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



def train(x_train, y_train, proj_dir, saved_model, batch_size, epoch, freeze_layer):

    """
    finetune a CNN model

    @params:
      saved_model   - required : saved CNN model for finetuning
      run_model     - required : CNN model name to be saved
      model_dir     - required : folder path to save model
      input_channel - required : model input image channel, usually 3
      freeze_layer  - required : number of layers to freeze in finetuning 
    
    """

    ## fine tune model
    pro_data_dir = os.path.join(proj_dir, 'pro_data')
    model = load_model(os.path.join(pro_data_dir, saved_model))
    ### freeze specific number of layers
    if freeze_layer != None:
        for layer in model.layers[0:freeze_layer]:
            layer.trainable = False
        for layer in model.layers:
            print(layer, layer.trainable)
    else:
        for layer in model.layers:
            layer.trainable = True
    #model.summary()

    ## fit data into dnn models
    history = model.fit(
        x=x_train,
        y=y_train,
        batch_size=batch_size,
        epochs=epoch,
        validation_data=None,
        verbose=1,
        callbacks=None,
        validation_split=0.3,
        shuffle=True,
        class_weight=None,
        sample_weight=None,
        initial_epoch=0)

    #### save final model
    model_fn = 'Tuned_model'
    model.save(os.path.join(pro_data_dir, model_fn))
    tuned_model = model
    print('fine tuning model complete!!')
    print('saved fine-tuned model as:', model_fn)
    y_pred = model.predict(x_test)
    y_pred_class = np.argmax(y_pred, axis=1)
    score = model.evaluate(x_test, y_test, verbose=0)
    loss = np.around(score[0], 3)
    acc = np.around(score[1], 3)
    print('acc:', acc)
    print('loss:', loss)
    # classification report
    #report = classification_report(y_test, y_pred_class, digits=3)
    #print(report)

    # save model
    model.save(os.path.join(pro_data_dir, 'invivo_model.h5'))

    # save a df for test and prediction
    df_test['y_pred'] = y_pred[:, 1]
    df_test['y_pred_class'] = y_pred_class
    df_test.rename(columns={'ROI_Class': 'y_test'}, inplace=True)
    test_voxel_pred = df_test[['Sub_ID', 'y_test', 'y_pred', 'y_pred_class']]
    test_voxel_pred.to_csv(os.path.join(pro_data_dir, 'invivo_voxel_pred.csv'))
    print('successfully save test voxel prediction!')

        cm = confusion_matrix(label, pred)
        cm_norm = cm.astype('float64') / cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.around(cm_norm, 2)
        # classification report
        report = classification_report(label, pred, digits=3)
        print(level)
        print(cm)
        print(cm_norm)
        print(report)

    if cm_type == 'norm':
        fmt = ''
    elif cm_type == 'raw':
        fmt = 'd'

    ax = sn.heatmap(
        cm0,
        annot=True,
        cbar=True,
        cbar_kws={'ticks': [-0.1]},
        annot_kws={'size': 26, 'fontweight': 'bold'},
        cmap='Blues',
        fmt=fmt,
        linewidths=0.5)

    ax.axhline(y=0, color='k', linewidth=4)
    ax.axhline(y=2, color='k', linewidth=4)
    ax.axvline(x=0, color='k', linewidth=4)
    ax.axvline(x=2, color='k', linewidth=4)

    ax.tick_params(direction='out', length=4, width=2, colors='k')
    ax.xaxis.set_ticks_position('top')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_aspect('equal')
    plt.tight_layout()

    fn = 'cm' + '_' + str(cm_type) + '_' + str(level) + '.png'
    plt.savefig(
        os.path.join(save_dir, fn),
        format='png',
        dpi=600)
    plt.close()
    return tuned_model


