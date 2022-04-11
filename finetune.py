
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



def finetune_exvivo(x_train, y_train, proj_dir, saved_model, batch_size, epoch, freeze_layer):

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
        initial_epoch=0
        )

    #### save final model
    model_fn = 'Tuned_model'
    model.save(os.path.join(pro_data_dir, model_fn))
    tuned_model = model
    print('fine tuning model complete!!')
    print('saved fine-tuned model as:', model_fn)

    return tuned_model


