import os
import numpy as np
import pandas as pd
import seaborn as sn
import glob2 as glob
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler

folds = 5

def get_data_invivo_kFoldVal(proj_dir, benign_bix, benign_nobix, pca_bix, exclude_patient,
                    exclude_list, x_input, folds):

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
    df1.columns = df1.columns.str.replace(' ', '')
    df2.columns = df2.columns.str.replace(' ', '')
    df3.columns = df3.columns.str.replace(' ', '')
    # print(df3.columns)
    df_exclude, df_excludeX, df_excludeY = [None, None, None]
    ## exlude 5 patients for validation
    if exclude_patient == True:
        print('exclude pat list:', exclude_list)
        indx = df3[df3['Sub_ID'].isin(exclude_list)].index
        print('total voxel:', df3.shape[0])
        df_exclude = df3.iloc[indx]
        # print(df_exclude.columns)
        df_excludeX = df_exclude.iloc[:, x_input]
        df_excludeY = df_exclude['ROI_Class'].astype('int')
        df_positions = df_exclude.loc[:,['Sub_ID','X','Y','Z']]
        # df_excludeX.to_csv(os.path.join(proj_dir, 'excludeX.csv'))
        # df_excludeY.to_csv(os.path.join(proj_dir, 'excludeY.csv'))
        # df_positions.to_csv(os.path.join(proj_dir, 'excludePositions.csv'))
        df3.drop(indx, inplace=True)
        print('total voxel:', df3.shape[0])
    else:
        df3 = df3

    ## split PCa cohorts to train/val and test
    train_inds, test_inds = next(GroupShuffleSplit(
        test_size=0.5,
        n_splits=2,
        random_state=7).split(df3, groups=df3['Sub_ID']))
    df3_train = df3.iloc[train_inds]
    df3_test = df3.iloc[test_inds]

    ## create train/val and test sets
    df_trainval = pd.concat([df1, df3_train])
    df_test = pd.concat([df2, df3_test])

    ## split PCa cohorts to train and val
    train_inds, val_inds = next(GroupShuffleSplit(
        test_size=0.2,
        n_splits=2,
        random_state=7).split(df_trainval, groups=df_trainval['Sub_ID']))
    df_train = df_trainval.iloc[train_inds]
    df_val = df_trainval.iloc[val_inds]
    # print(df_trainval.shape)
    # trainValIdxs = np.range(df_trainval.shape[0])
    # trainValIdxs = random.shuffle(trainValIdxs)
    # allTrainValParts = [None] * folds
    # trainIdxs  = [None] * folds
    # for i in range(folds):
    #     allTrainValParts[i] = trainValIdxs[trainValIdxs % folds == i]
    #     trainIdxs[i] = trainValIdxs[trainValIdxs % folds != i]

    # valIdxs = allTrainValParts

    # get data and labels
    x_train = df_train.iloc[:, x_input]
    y_train = df_train['ROI_Class'].astype('int')
    x_val = df_val.iloc[:, x_input]
    y_val = df_val['ROI_Class'].astype('int')
    x_test = df_test.iloc[:, x_input]
    y_test = df_test['ROI_Class'].astype('int')

    trainIDs = df_train.loc[:, 'Sub_ID'].to_numpy()
    valIDs = df_val.loc[:, 'Sub_ID'].to_numpy()
    testIDs = df_test.loc[:, 'Sub_ID'].to_numpy()

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

    return x_train, y_train, x_val, y_val, x_test, y_test, df_val, df_test, trainIDs, valIDs, testIDs, df_excludeX, df_excludeY, df_positions