import os
import numpy as np
import pandas as pd
import seaborn as sn
import glob2 as glob
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from imblearn.over_sampling import SMOTE


def get_data(data_dir, x_input):

    """
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
    """

    df = pd.read_csv(data_dir + '/' + 'invivo_grade.csv')
    #df.iloc[:, 1:] = (data.iloc[:, 1:])/(data.iloc[:, 1:].max())
    df.loc[df.loc['ROI Name'] == 1] = 0
    df.loc[df.loc['ROI Name'] == 2] = 1
    df.loc[df.loc['ROI Name'] == 3] = 2
    df.loc[df.loc['ROI Name'] == 4] = 3
    df.loc[df.loc['ROI Name'] == 5] = 4

    # data split
    X = df.iloc[:, self.x_input]
    Y = df.y_cat.astype('int')
    x_train, x_test_1, y_train, y_test_1 = train_test_split(
        X, Y, test_size=0.3, random_state=1234)
    x_val, x_test, y_val, y_test = train_test_split(
        x_test_1, y_test_1, test_size=0.3, random_state=1234)

    ## oversample
    resample = SMOTE(random_state=42)
    x_train, y_train = resample.fit_resample(x_train, y_train)
    x_val, y_val = resample.fit_resample(x_val, y_val)
    print('train:', y_train.count())
    print('val:', y_val.count())

    # scale data#
    x_train = MinMaxScaler().fit_transform(x_train)
    x_val = MinMaxScaler().fit_transform(x_val)
    x_test = MinMaxScaler().fit_transform(x_test)

    pd.set_option('display.max_columns', 500)
    pd.set_option('display.max_rows', 500)
    print('train size:', len(y_train))
    print('val size:', len(y_val))
    print('test size:', len(y_test))

    return x_train, x_val, x_test, y_train, y_val, y_test




