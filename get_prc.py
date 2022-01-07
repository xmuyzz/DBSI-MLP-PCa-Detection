import os
import numpy as np
import pandas as pd
from utils.plot_prc import plot_prc
from utils.roc_bootstrap import roc_bootstrap


def get_prc(proj_dir, output_dir, data_type):
    
    pro_data_dir = os.path.join(proj_dir, 'pro_data')

    if data_type == 'invivo':
        fn1 = 'invivo_voxel_pred.csv'
        fn2 = 'invivo_pat_pred.csv'
    elif data_type == 'exvivo':
        fn1 = 'exvivo_voxel_pred.csv'
        fn2 = 'exvivo_pat_pred.csv'

    # voxel level data
    df1 = pd.read_csv(os.path.join(pro_data_dir, 'test_voxel_pred.csv'))
    y_test_vox = df1['y_test'].to_numpy()
    y_pred_vox = df1['y_pred'].to_numpy()
    
    # patient level data
    df2 = pd.read_csv(os.path.join(pro_data_dir, 'test_pat_pred.csv'))
    y_test_pat = df2['y_test'].to_numpy()
    y_pred_pat = df2['y_pred'].to_numpy()

    ## ROC
    preds = [y_pred_vox, y_pred_pat]
    labels = [y_test_vox, y_test_pat]
    level = ['voxel', 'patient']
    
    for pred, label, level in zip(preds, labels, level):
        
        prc_auc = plot_prc(
            save_dir=output_dir, 
            y_true=label, 
            y_pred=pred, 
            level=level, 
            color='red',
            data_type=data_type
            )
    
        print(level, prc_auc)

    return prc_auc
