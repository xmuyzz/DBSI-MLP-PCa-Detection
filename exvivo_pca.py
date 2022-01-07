
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

# ----------------------------------------------------------------------------------
# construct train, validation and test dataset
# ----------------------------------------------------------------------------------
def get_data(proj_dir, data, exclude_pat, x_input):
    
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
    
    ## load data     
    df = pd.read_csv(os.path.join(proj_dir, data))
    df['ID'] = df['Sub_ID'] + df['ROI_Class']
    df['ROI_Class'].replace(['BPZ', 'BPH', 'SBPH', 'PCa'], [3, 3, 0, 1], inplace=True)
    df.fillna(0, inplace=True)
    ## only include class 0 and class 1
    df = df[df['ROI_Class'].isin([0, 1])]
    ## exlude patients
    if exclude_pat != None:
        indx = df[df['ID'].isin(exclude_pat)].index
        print('total voxel:', df.shape[0])
        df.drop(indx, inplace=True)
        print('total voxel:', df.shape[0])
    
    #----------------------
    ## if wu_split == True
    #----------------------
    ## split df into CH and WU cohorts
    dfSH = df.iloc[0 :89288]
    dfWU = df.iloc[89288:]
    #dfWU = pd.DataFrame(data=dfWU.values, columns=dfSH.columns)
    print('dfSH:', dfSH[0:10])
    print('dfWU:', dfWU[0:10])

    ## split PCa cohorts to train and test
    train_inds, test_inds = next(GroupShuffleSplit(
        test_size=0.3,
        n_splits=2, 
        random_state=0).split(dfWU, groups=dfWU['ID']))
    df_train = dfWU.iloc[train_inds]
    df_test1 = dfWU.iloc[test_inds]
    df_test2 = dfSH
    
    # get data and labels
    x_train = df_train.iloc[:, x_input]
    y_train = df_train['ROI_Class'].astype('int')
    x_test1 = df_test1.iloc[:, x_input]
    y_test1 = df_test1['ROI_Class'].astype('int')
    x_test2 = dfSH.iloc[:, x_input]
    y_test2 = dfSH['ROI_Class'].astype('int')

    ## oversample
    resample = SMOTE(random_state=42)
    x_train, y_train = resample.fit_resample(x_train, y_train)
    print(y_train.count())
    print(y_train[0:100])

    # scale data#
    x_train = MinMaxScaler().fit_transform(x_train)
    x_test1 = MinMaxScaler().fit_transform(x_test1)
    x_test2 = MinMaxScaler().fit_transform(x_test1)
   
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.max_rows', 500)
    #print(y_val[4500:5000])
    print('train size:', len(y_train))
    print('test1 size:', len(y_test1))
    print('test2 size:', len(y_test2))
    
    #----------------------
    # if wu_split == False
    #----------------------
    train_inds, test_inds = next(GroupShuffleSplit(
        test_size=0.3,
        n_splits=2,
        random_state=42).split(df, groups=df['ID']))
    df_train = df.iloc[train_inds]
    df_test = df.iloc[test_inds]

    # get data and labels
    x_train = df_train.iloc[:, x_input]
    y_train = df_train['ROI_Class'].astype('int')
    x_test = df_test.iloc[:, x_input]
    y_test = df_test['ROI_Class'].astype('int')

    ## oversample
    resample = SMOTE(random_state=42)
    x_train, y_train = resample.fit_resample(x_train, y_train)
    print('df count:', y_train.count())

    # scale data#
    x_train = MinMaxScaler().fit_transform(x_train)
    x_test = MinMaxScaler().fit_transform(x_test)

    pd.set_option('display.max_columns', 500)
    pd.set_option('display.max_rows', 500)
    #print(y_val[4500:5000])
    print('train size:', len(y_train))
    print('test size:', len(y_test))

    return x_train, y_train, x_test, y_test, df_test, x_test1, y_test1, \
           x_test2, y_test2, df_test1, df_test2

# ----------------------------------------------------------------------------------
# construct DNN model with batch normalization layers and dropout layers
# ----------------------------------------------------------------------------------
def finetune_model(x_train, y_train, output_dir, saved_model, batch_size, epoch, freeze_layer):

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
    model = load_model(os.path.join(output_dir, saved_model))
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
    model.save(os.path.join(output_dir, model_fn))
    tuned_model = model
    print('fine tuning model complete!!')
    print('saved fine-tuned model as:', model_fn)

    return tuned_model


# ----------------------------------------------------------------------------------
# trainning DNN model
# ----------------------------------------------------------------------------------
def model_test(tuned_model, wu_split, x_test, y_test, df_test, x_test1, y_test1, 
               x_test2, y_test2, df_test1, df_test2):
    
    if wu_split == True:
        x_test = x_test1
        y_test = y_test1
        df_test = df_test1
    else:
        x_test = x_test
        y_test = y_test
        df_test = df_test

    y_pred = tuned_model.predict(x_test)  
    y_pred_class = np.argmax(y_pred, axis=1)
    score = tuned_model.evaluate(x_test, y_test, verbose=0)
    loss = np.around(score[0], 3)
    acc = np.around(score[1], 3)
    print('acc:', acc)
    print('loss:', loss)
    
    ### save a df for test and prediction
    df_test['y_pred'] = y_pred[:, 1]
    df_test['y_pred_class'] = y_pred_class
    test_voxel_pred = df_test[['ID', 'ROI_Class', 'y_pred', 'y_pred_class']]
    test_voxel_pred.to_csv(os.path.join(output_dir, 'exvivo_test_voxel_pred.csv'))
    
    ## patient level
    df_mean = test_voxel_pred.groupby(['ID'], as_index=False).mean()
    print(df_mean)
    label_pat = df_mean['ROI_Class'].to_numpy()
    pred_pat = df_mean['y_pred'].to_numpy()
    print(label_pat)
    print(pred_pat)
    # get pred class on patient level
    pred_class_pat = []
    for pred in pred_pat:
        if pred > 0.2:
            pred = 1
        else:
            pred = 0
        pred_class_pat.append(pred)
    pred_class_pat = np.asarray(pred_class_pat)
    df_mean['pred_class_pat'] = pred_class_pat
    df_mean['label_pat'] = label_pat
    df_mean['pred_pat'] = pred_pat
    test_pat_pred = df_mean[['ID', 'label_pat', 'pred_pat', 'pred_class_pat']]
    print(test_pat_pred)
    print('patient lable:', test_pat_pred.groupby('label_pat').count())
    test_pat_pred.to_csv(os.path.join(output_dir, 'test_pat_pred.csv'))
    
    # get confusiom matrix    
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
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        threshold = dict()
        fpr, tpr, threshold = roc_curve(label, pred)
        roc_auc = np.around(auc(fpr, tpr), 3)
        print('prediction level:', level)
        print('ROC AUC:', roc_auc)

    return loss, acc


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

    proj_dir = '/home/xmuyzz/Harvard_AIM/others/pca/data/'
    output_dir  = '/home/xmuyzz/Harvard_AIM/others/pca/output/'
    data = 'pca_exvivo.csv'
    saved_model = 'saved_model.h5'
    wu_split = False 
    exclude_pat = None
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
    batch_size = 128
    lr = 0.001
    epoch = 10
    freeze_layer = None
    init = 'he_uniform' 
    optimizer = Adam(learning_rate=lr)
    loss = 'sparse_categorical_crossentropy'
    output_activation = 'softmax'
    activation = ELU(alpha=ELU_alpha)     
    
    '''
    keranl initializer: 'he_uniform', 'lecun_normal', 'lecun_uniform'
    optimizer function: 'adam', 'adamax', 'nadam', 'sgd'
    loss function: 'categorical_crossentropy'
    activation function: LeakyReLU(alpha=alpha)
    '''

#    parser = argparse.ArgumentParser(description='PCa ex vivo')
#    
#    parser.add_argument('data_dir', help='Directory in which test data are stored')
#    parser.add_argument('model_dir', type=str, help='folder of trained model weight')
#    parser.add_argument('--ct_format', type=str, default='nifti', 
#                        help='test data format,nrrd/nii.gz are feasible')
#    parser.add_argument('--labels_ready','-l', type=bool, default=True, 
#                        help='check if test images has corresponding labels') 
#    parser.add_argument('--load_weights','-w',type=str, default= 'load2test.pth', 
#                        help='load weights to initialise model')
#    parser.add_argument('--inter_spacing','-s', type=list, default=[0.7,0.7,2.5], 
#                        help='spacing of image_preprocessing')
#    parser.add_argument('--hu_window','-hu', type=list, default=[-200,250], help='hu_window')
#    parser.add_argument('--plot_dice_score','-p', type=bool, default=True, 
#                        help='plot_dice_score into histogram')
#    args = parser.parse_args()
    
    

    start = timeit.default_timer()
    
    x_train, y_train, x_test, y_test, df_test, x_test1, \
    y_test1, x_test2, y_test2, df_test1, df_test2 = get_data(
        proj_dir=proj_dir, 
        data=data,
        exclude_pat=exclude_pat, 
        x_input=x_input
        )
    
    tuned_model = finetune_model(
        x_train=x_train,
        y_train=y_train,
        output_dir=output_dir,
        saved_model=saved_model,
        batch_size=batch_size,
        epoch=epoch,
        freeze_layer=freeze_layer
        )
    
    loss, acc = model_test(
        tuned_model=tuned_model,
        wu_split=wu_split,
        x_test=x_test,
        y_test=y_test,
        df_test=df_test,
        x_test1=x_test1,
        y_test1=y_test1,
        x_test2=x_test2,
        y_test2=y_test2,
        df_test1=df_test1,
        df_test2=df_test2
        )
    
    #plot_CM(cm_norm, '')

    stop = timeit.default_timer()
    running_seconds = np.around(stop - start, 0)
    running_minutes = np.around(running_seconds/60, 0)
    print('\nDNN Running Time:', running_minutes, 'minutes')



