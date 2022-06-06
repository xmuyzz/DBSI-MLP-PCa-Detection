
mport os
import numpy as np
import pandas as pd
import nibabel as nib
import ntpath
import queue
import datetime
import matplotlib.pyplot as plt
from functools import partial
from tensorflow.keras.models import load_model
import glob



def get_roi_data(data_dir, out_dir):

    param_list = [
                 'b0_map',                        #07
                 'dti_adc_map',                   #08
                 'dti_axial_map',                 #09
                 'dti_fa_map',                    #10
                 'dti_radial_map',                #11
                 'fiber_ratio_map',            #12
                 'fiber1_axial_map',              #13
                 'fiber1_fa_map',                 #14
                 'fiber1_fiber_ratio_map',     #15
                 'fiber1_radial_map',             #16
                 'fiber2_axial_map',              #17
                 'fiber2_fa_map',                 #18
                 'fiber2_fiber_ratio_map',     #19
                 'fiber2_radial_map',             #20
                 'hindered_adc_map',              #21
                 'hindered_ratio_map',         #22
                 'iso_adc_map',                   #23
                 'restricted_adc_1_map',     #24
                 'restricted_adc_2_map',            #25
                 'restricted_ratio_1_map',#26
                 'restricted_ratio_2_map',       #27
                 'water_adc_map',                 #28
                 'water_ratio_map',            #29
                ]
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

    dirs = []
    for dirName, subdirList, fileList in os.walk(data_dir):
        for dirname in range(len(subdirList)):
            if subdirList[dirname] == 'DBSI_results_0.1_0.1_0.8_0.8_1.5_1.5':
                #look for correct thresholds of files; append to dirs
                dirs.append(os.path.join(dirName, subdirList[dirname]))  
    # insert voxel index and information into each column
    col_list = param_id  
    col_list.insert(0, 'ROI_Class')
    col_list.insert(0, 'ROI_ID')
    col_list.insert(0, 'Voxel')
    col_list.insert(0, 'Z')
    col_list.insert(0, 'Y')
    col_list.insert(0, 'X')
    col_list.insert(0, 'Sub_ID')
    df_stat = pd.DataFrame([], columns=col_list)
    for dir in dirs: 
        sub_id = os.path.basename(os.path.dirname(dir))   
        #print(sub_id)        
        roi_path = dir
        rois = [os.path.join(roi_path, 'roi.nii.gz')]
        # looking at each roi individually                       
        for roi in rois: 
            stat = []
            try:
                atlas = nib.load(roi).get_data() 
            except:
                print('No roi')
                continue
            roi_folder, roi_name = os.path.split(roi) 
            current_dir = dir
            # find all rois per file, look at the first one
            if len(np.unique(atlas[atlas > 0])) > 0: 
                roi_id = np.unique(atlas[atlas > 0])[0]
            idx = np.asarray(np.where(atlas == roi_id))
            for item in range(len(param_list)):
                print(item)
                print(param_list[item])
                img = nib.load(glob.glob(os.path.join(current_dir, param_list[item] + '.nii'))[0]).get_data()
                #print(img)
                sub_data = img[atlas == roi_id]
                stat.append(sub_data)
            # insert voxel index and information into each column  
            val = np.asarray(stat).astype(np.float32)
            # -4 for .nii file, -7 for nii.gz file
            val = np.concatenate((np.repeat(roi_name[:-7], len(sub_data))[np.newaxis], val), axis=0)
            val = np.concatenate((np.repeat(roi_id, len(sub_data))[np.newaxis], val), axis=0)
            val = np.concatenate((np.asarray(range(0, len(sub_data)))[np.newaxis], val), axis=0)
            val = np.concatenate((idx, val), axis=0)
            val = np.concatenate((np.repeat(sub_id, len(sub_data))[np.newaxis], val), axis=0)
            val = np.transpose(val)
            df = pd.DataFrame(val, columns=col_list)
            df_stat = pd.concat([df_stat, df])
            df_stat[df_stat.columns[7:29]] = df_stat[df_stat.columns[7:29]].astype('float64')
    df_stat.fillna(df_stat.median(), inplace=True)
    csv_file = 'prostate' + '.csv'
    df_stat.to_csv(os.path.join(out_dir, csv_file), index=False)
    print(df_stat)

    return df_stat


def model_predict(pro_data_dir, out_dir, df_stat, x_input, overlaid_map):

    df = df_stat
    x_pred  = df.iloc[:, x_input].astype('float64')
    model = load_model(os.path.join(pro_data_dir, 'Tuned_model'))
    pred = model.predict(x_pred)
    print(pred)
    pred_class = np.argmax(pred, axis=1)
    print(pred_class)
    img = np.zeros(shape=(128, 128, 10))
    x_index = np.asarray(df.iloc[:, [1]])[:, 0].astype(int)
    y_index = np.asarray(df.iloc[:, [2]])[:, 0].astype(int)
    z_index = np.asarray(df.iloc[:, [3]])[:, 0].astype(int)
    for i in range(x_index.shape[0]):
        img[x_index[i], y_index[i], z_index[i]] = pred_class[i]
    aff = nib.load(overlaid_map).affine
    tumor_pred = nib.Nifti1Image(img, aff)
    nib.save(tumor_pred, os.path.join(out_dir, 'tumor_map.nii.gz'))

    return tumor_pred


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

    tumor_pred = model_predict()
    #roi = PCa_pred
    roi = os.path.join(pred_dir, 'tumor_pred.nii.gz')
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


if __name__ == '__main__':

    
    x_input = range(12, 30)
    conn = 0 # if conn=0, 8-connectivity; if 1, then 4-connectivity
    threshold = 5 # threshold of elimination (component size)
    data_dir = '/mnt/aertslab/USERS/Zezhong/others/pca/exvivo/WU007F11'
    out_dir = '/mnt/aertslab/USERS/Zezhong/others/pca/exvivo'
    pro_data_dir = '/home/xmuyzz/Harvard_AIM/others/pca/pro_data'
    overlaid_map = os.path.join(data_dir, 'DBSI_results_0.1_0.1_0.8_0.8_1.5_1.5/dti_adc_map.nii')
 
    np.set_printoptions(threshold=np.inf)
    df_stat = get_roi_data(data_dir, out_dir)
    tumor_pred = model_predict(pro_data_dir, out_dir, df_stat, x_input, overlaid_map)

    #filtered_map = tumor_map_filter()









