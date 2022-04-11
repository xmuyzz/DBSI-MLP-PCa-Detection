import os
import numpy as np
import pandas as pd
import seaborn as sn
import glob2 as glob
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler



def get_data_invivo(proj_dir, benign_bix, benign_nobix, pca_bix, exclude_patient,
                    exclude_list, x_input):

    """
    get in vivo dataset

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
         'water_ratio_map.nii']              #29
        
    """

    data_dir = os.path.join(proj_dir, 'data')

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


def get_data_exvivo(proj_dir, exvivo_data, exclude_list, x_input):

    """
    get in vivo dataset

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
         'water_ratio_map.nii']              #29
    
    """

    ## load data
    data_dir = os.path.join(proj_dir, 'data')
    df = pd.read_csv(os.path.join(data_dir, exvivo_data))
    df['ID'] = df['Sub_ID'] + df['ROI_Class']
    df['ROI_Class'].replace(['BPZ', 'BPH', 'SBPH', 'PCa'], [0, 0, 0, 1], inplace=True)
    df.fillna(0, inplace=True)
    ## only include class 0 and class 1
    df = df[df['ROI_Class'].isin([0, 1])]
    ## exlude patients
    if exclude_list != None:
        print(exclude_list)
        indx = df[df['ID'].isin(exclude_list)].index
        print('total voxel:', df.shape[0])
        df.drop(indx, inplace=True)
        print('total voxel:', df.shape[0])
    else:
        print('no exlcude patients!')

    ## if wu_split == True
    #----------------------
    ## split df into CH and WU cohorts
    dfSH = df.iloc[0 :89288]
    dfWU = df.iloc[89288:]
    #dfWU = pd.DataFrame(data=dfWU.values, columns=dfSH.columns)
    print('dfSH:', dfSH[0:10])
    print('dfWU:', dfWU[0:10])

    ## split PCa cohorts to train and test
    train_inds, test_inds = next(
        GroupShuffleSplit(
        test_size=0.3,
        n_splits=2,
        random_state=0
        ).split(dfWU, groups=dfWU['ID'])
        )
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






