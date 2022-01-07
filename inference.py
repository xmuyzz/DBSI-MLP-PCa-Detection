#----------------------------------------------------------------------
# deep learning classifier using a multiple layer perceptron (MLP)
# 
# Author: Zezhong Ye @ Washington University School of Medicine
# Date: 03.14.2019
# Contact: ze-zhong@wustl.edu 
#-------------------------------------------------------------------------------------------


import os
import winsound
import timeit
import itertools
import numpy as np
import pandas as pd
import seaborn as sn
import glob2 as glob
import nibabel as nib
import ntpath
import queue
import datetime
import matplotlib.pyplot as plt
from functools import partial
from datetime import datetime
from collections import Counter
from time import gmtime, strftime

from imblearn.over_sampling import SMOTE, SVMSMOTE, KMeansSMOTE
from imblearn.over_sampling import RandomOverSampler

import tensorflow
from tensorflow.keras import initializers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Input, Dense, Reshape, Activation, Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.activations import elu, relu, softmax
from tensorflow.keras.initializers import HeNormal

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve, auc
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MaxAbsScaler, StandardScaler, MinMaxScaler 



# load inference data
# --------------------------      
def load_pred_data():

    # DBSI metric map list for training data
    param_id = [
                 'b0',                        #07
                 'dti_adc',                   #08
                 'dti_axial',                 #09
                 'dti_fa',                    #10
                 'dti_radial',                #11
                 'fiber_fraction',            #12
                 'fiber1_axial',              #13
                 'fiber1_fa',                 #14
                 'fiber1_fiber_fraction',     #15
                 'fiber1_radial',             #16
                 'fiber2_axial',              #17
                 'fiber2_fa',                 #18
                 'fiber2_fiber_fraction',     #19
                 'fiber2_radial',             #20
                 'hindered_adc',              #21
                 'hindered_fraction',         #22
                 'iso_adc',                   #23
                 'highly_restricted_adc',     #24
                 'restricted_adc',            #25
                 'highly_restricted_fraction',#26
                 'restricted_fraction',       #27
                 'water_adc',                 #28
                 'water_fraction',            #29
                ]
   
    pred_file = str(case_id) + '_' + 'prostate' + '.csv'
    df_pred = pd.read_csv(os.path.join(pred_dir, pred_file))
    x_pred  = df_pred.iloc[:, x_input].astype('float64')

    return x_pred, df_pred


# prediction
# ----------------
def model_predict():

    img = np.zeros(shape=(280, 224, 24))
    x_index = np.asarray(df_pred.iloc[:, [1]])[:, 0].astype(int)
    y_index = np.asarray(df_pred.iloc[:, [2]])[:, 0].astype(int)
    z_index = np.asarray(df_pred.iloc[:, [3]])[:, 0].astype(int)

    for i in range(x_index.shape[0]):
        img[x_index[i], y_index[i], z_index[i]] = pred_class[i]
            
    aff = nib.load(
        os.path.join(pred_dir, DBSI_folder, overlaid_map)
        ).get_affine()
    
    PCa_pred = nib.Nifti1Image(img, aff)
    PCa_pred_map = 'tumor_map' + '_' + strftime('%d-%b-%Y-%H-%M-%S', gmtime()) + '.nii'
    #pred_map = 'tumor_map.nii'
    nib.save(PCa_pred, os.path.join(pred_dir, PCa_pred_map))

    return PCa_pred_map, PCa_pred


# prediction map filter
# ------------------------------
def flood(img, position, conn):
    
    floodMask = np.zeros(img.shape)
    edge = queue.Queue()
    visited = np.zeros(img.shape)
    dim0 = img.shape[0]
    dim1 = img.shape[1]
    connectSet = []
    
    if conn==0:
        connectSet = [[1, -1], [1, 0], [1, 1], [0, -1], [0, 1], [-1, -1], [-1, 0], [-1, 1]]
    elif conn==1:
        connectSet = [[0, 1], [1, 0], [-1, 0], [0, -1]]
    else:
        print('conn is not valid. Please enter 0 or 1 for 8-connectivity and 4-connectivity respectively')
        return None
    
    edge.put(position)
    visited[position[0], position[1]] = 1
    
    flag = False
    while edge.empty() == False:
        expandPoint = edge.get()
        if img[expandPoint[0], expandPoint[1]] > 0:
            floodMask[expandPoint[0], expandPoint[1]] = 1
            # print("1edge: ",expandPoint,"; value: ", img[expandPoint[0],expandPoint[1]])
            for adjacent in connectSet:
                # print(adjacent)
                edgePoint = np.add(expandPoint, adjacent)
                # print("edge: ",edgePoint,"; value: ", img[edgePoint[0],edgePoint[1]],0 <= edgePoint[0] <= dim0-1,0 <= edgePoint[1] <= dim1-1,0 <= edgePoint[0] <= dim0-1 and 0 <= edgePoint[1] <= dim1-1)
                if 0 <= edgePoint[0] <= dim0-1 and 0 <= edgePoint[1] <= dim1-1:
                    # print("edge: ",edgePoint,"; value: ", img[edgePoint[0],edgePoint[1]],visited[edgePoint[0], edgePoint[1]])
                    if visited[edgePoint[0], edgePoint[1]] == 0 and img[edgePoint[0], edgePoint[1]] > 0:
                        # print("edge: ",edgePoint,"; value: ", img[edgePoint[0],edgePoint[1]])
                        floodMask[edgePoint[0], edgePoint[1]] = 1
                        visited[edgePoint[0], edgePoint[1]] = 1
                        edge.put(edgePoint)
                        flag = True
                    else:
                        visited[edgePoint[0], edgePoint[1]] = 1
                        # print("out of bounds or nothing there")
    # if flag == True:
    #     print("sum floodMask:",sum(floodMask.flatten()))
    return floodMask

def labelComponents(img, conn): 
    #for each voxel, put size of the component it is connected to on it
    
    binaryLabels = np.zeros(img.shape)
    dim0 = img.shape[0]
    dim1 = img.shape[1]
    
    for i in range(dim0):    
        for j in range(dim1):
            binaryLabels = np.add(binaryLabels, flood(img, [i, j], conn))
    # print(binaryLabels)
    return binaryLabels

def getComponentsThreshold(img, conn, threshold):
    
    componentsAboveThresh = np.zeros(img.shape)
    dim0 = img.shape[0]
    dim1 = img.shape[1]
    componentLabels = labelComponents(img, conn)
    #print(np.unique(componentLabels))
    
    for i in range(dim0):
        for j in range(dim1):
            if componentLabels[i, j] > threshold:
                componentsAboveThresh[i, j] = 1
                
    return componentsAboveThresh

def tumor_map_filter():

    PCa_pred_map, PCa_pred = model_predict()
    #roi = PCa_pred
    roi = os.path.join(pred_dir, PCa_pred_map)、
    try:
        atlas = nib.load(roi).get_data()
    except:
        print('No roi')
    roiArr = np.asarray(atlas)
    #print(roiArr.shape)
    layersWithROI = [] #records the layer number of layers with ROIs marked
    
    for x in range(roiArr.shape[2]):
        if sum(roiArr[:, :, x].flatten()) > 0:
            layersWithROI.append(x)        
    filteredROI = np.zeros(roiArr.shape)

    for layer in layersWithROI:
        filteredROI[:, :, layer] = getComponentsThreshold(roiArr[:, :, layer], conn, threshold)

    #TODO: save filteredROI as .nii file
    aff = nib.load(roi).get_affine()
    filtered_map = nib.Nifti1Image(filteredROI, aff)

    #filename = roi[:-4]+ '_' + 'filtered' + '_' + strftime('%d-%b-%Y-%H-%M-%S', gmtime()) + '.nii'
    filename = roi[:-4] + '_' + 'filtered' + '_' + str(threshold) + '.nii'
    nib.save(filtered_map, filename)

    return filtered_map

    # np.savetxt(roi_path+"test.csv", labelComponents(roiArr[:,:,9], conn), delimiter=',')
    # print(sum(flood(roiArr[:,:,9], [149,104], conn).flatten()))


# ----------------------------------------------------------------------------------
# run the model
# ---------------------------------------------------------------------------------- 
if __name__ == '__main__':

    
    x_input = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 21, 22, 23, 24, 25, 26, 28, 29]
    conn = 0 # if conn=0, 8-connectivity; if 1, then 4-connectivity
    threshold = 5 # threshold of elimination (component size)

    proj_dir = 'D:/2020_PCa_Project/WUSM_PCa/DBSI_Data'
    overlaid_map = 'dti_adc_map.nii'
    ROI_name = 'roi.nii.gz'
    DBSI_folder = 'DHI_results_0.1_0.8_0.8_2.3_2.3'

    list_case = ['P011', 'P012', 'P013', 'P014', 'P015', 'P016', 'P017', 'P019',
                 'P021', 'P022', 'P023', 'P025', 'P026', 'P028', 'P030']

    #list_case = ['P023']

    tot_run = len(list_case)
    
    np.set_printoptions(threshold=np.inf)

    start = timeit.default_timer()

    for case_id in list_case:

        count += 1
        print('\nRun:' + str(count) + '/' + str(tot_run))

        print(str(case_id))

        pred_dir = os.path.join(proj_dir, case_id)

        x_pred, df_pred = load_pred_data()

        model = load_model(os.path.join(pred_dir, 'my_model'))

        pred = model.predict(x_pred)
        pred_class = np.argmax(pred, axis=1)

        PCa_pred_map, PCa_pred = model_predict()

        filtered_map = tumor_map_filter()


    stop = timeit.default_timer()
    print('DNN Running Time:', np.around(stop - start, 0), 'seconds')

    # dound alarm
    duration = 200                       # milliseconds
    freq     = 440                       # Hz
    winsound.Beep(freq, duration)









##
### ----------------------------------------------------------------------------------
### create prediction
### ----------------------------------------------------------------------------------
##def show_map():
##
##    atlas = pred_map.astype(float)
##    co = coordinates - 1
##    
##    img_adc = np.zeros([atlas.shape[0],atlas.shape[1],atlas.shape[2]])
##    img_b0  = np.zeros([atlas.shape[0],atlas.shape[1],atlas.shape[2]])
##    
##    temp_atlas = copy.deepcopy(atlas)
##    temp_atlas[temp_atlas[:]==0] = np.nan
##    
##    adc_data = np.asarray(adc_data['DTI_ADC']).transpose()
##    b0_data  = np.asarray(adc_data['DTI_FA']).transpose()
##    
##    for img_idx in range(co.shape[1]):
##        
##        imdx = list(co[:, img_idx].astype(int))
##        img_adc[imdx[0],imdx[1],imdx[2]] = adc_data[0,img_idx]
##        img_b0[imdx[0],imdx[1],imdx[2]]  = b0_data[0,img_idx]
##        
##    fig = plt.figure(figsize=(24, 15))
##    #ax  = fig.add_subplot(1,1,1)
##    ax_1 = fig.add_subplot(2, 1, 1)
##    ax.axis('off')
##    ax.imshow(img_adc[:, :, idx[2, 1]], cmap='gray', vmin=0.2, vmax=3.0, aspect='equal')
##    ax.imshow(temp_atlas[:, :, idx[2,1]], cmap='rainbow', alpha=0.35, aspect='equal')
##    ax.set_title('Tumor Prediction', fontweight='bold')
##    
##    ax_2 = fig.add_subplot(2, 2, 1)
##    ax_2.axis('off')
##    ax_2.set_title('b0', fontweight='bold')
##    ax_2.imshow(img_b0[:, :, idx[2,1]], cmap='gray', vmin=0, vmax=10, aspect='equal')
##    ax_2.imshow(temp_atlas[:, :, idx[2,1]], cmap='rainbow', alpha=0.35, aspect='equal')
##    
##    plt.savefig(os.path.join(result_dir, 'tumor_map_%s.PNG'%roi_name[:-4]), format='PNG', dpi=100)
##    plt.show()
##    plt.close()
