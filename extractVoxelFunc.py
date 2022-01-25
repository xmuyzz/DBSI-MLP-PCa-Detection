import os
import nibabel as nib
import numpy as np
import pandas as pd
import glob
import ntpath

# caseName = 'Vic_2018_12_22'
# roiName = 'PCa.nii.gz'
#data folder currently: r'Y:\Zezhong_Ye\Prostate_Cancer_Project_Shanghai\ALL_DBSI_DATA_new_thresh_all\PCa_with_biopsy'
# thresholds = [0.1, 0.8,2.3]
def extractVoxelFunc(dataFolder, saveFolder, caseName, roiName, thresholds):
    dirs = []
    patientFolder = os.path.join(dataFolder, caseName)
    print("extract voxel values and index from ROI and DBSI data: start...")
    for dirName, subdirList, fileList in os.walk(patientFolder):
        for dirname in range(len(subdirList)):
            if subdirList[dirname] == 'DBSI_results_%s_%s_%s_%s_%s_%s' % (
                                                                          thresholds[0],
                                                                          thresholds[0],
                                                                          thresholds[1],
                                                                          thresholds[1],
                                                                          thresholds[2],
                                                                          thresholds[2]):
                                                                          dirs.append(os.path.join(dirName,
                                                                                                   subdirList[dirname])
                                                                          )
                
                # if subdirList[dirname] == 'DHISTO_results_%s_%s_%s_%s_%s_%s' % (
                #                                                             thresholds[0],
                #                                                             thresholds[0],
                #                                                             thresholds[1],
                #                                                             thresholds[1],
                #                                                             thresholds[2],
                #                                                             thresholds[2]):
                #                                                             dirs.append(os.path.join(dirName,
                #                                                                                      subdirList[dirname])
                #                                                             )
    # DBSI parameters from data file
    print(dirs)
    param_list = [
                  'b0_map',
                  'dti_adc_map',
                  'dti_axial_map',
                  'dti_fa_map',
                  'dti_radial_map',
                  'fiber_ratio_map',
                  'fiber1_axial_map',
                  'fiber1_fa_map',
                  'fiber1_fiber_ratio_map',
                  'fiber1_radial_map',
                  'fiber2_axial_map',
                  'fiber2_fa_map',
                  'fiber2_fiber_ratio_map',
                  'fiber2_radial_map',
                  'hindered_adc_map',
                  'hindered_ratio_map',
                  'iso_adc_map',
                  'restricted_adc_1_map',
                  'restricted_adc_2_map',
                  'restricted_ratio_1_map',
                  'restricted_ratio_2_map',
                  'water_adc_map',
                  'water_ratio_map'
                 ]

    # ID for different DBSI metric maps
    param_id = [
                'b0',
                'dti_adc',
                'dti_axial',
                'dti_fa',
                'dti_radial',
                'fiber_ratio',
                'fiber1_axial',
                'fiber1_fa',
                'fiber1_fiber_ratio',
                'fiber1_radial',
                'fiber2_axial',
                'fiber2_fa',
                'fiber2_fiber_ratio',
                'fiber2_radial',
                'hindered_adc',
                'hindered_ratio',
                'iso_adc',
                'restricted_adc_1',
                'restricted_adc_2',
                'restricted_ratio_1',
                'restricted_ratio_2',
                'water_adc',
                'water_ratio'
               ]

    # insert voxel index and information into each column
    col_list = param_id
    col_list.insert(0, "ROI_Class")
    col_list.insert(0, "ROI_ID")
    col_list.insert(0, "Voxel")
    col_list.insert(0, "Z")
    col_list.insert(0, "Y")
    col_list.insert(0, "X")
    col_list.insert(0, "Sub_ID")
    stat_df = pd.DataFrame([], columns=col_list)
    print(dirs)
    for dir in dirs:
        sub_id = os.path.basename(os.path.dirname(dir))
        print(sub_id)
        roi_path = dir

        # different PCa tissue types
        # rois = [os.path.join(roi_path, roiName + '.nii')]
        rois = [os.path.join(roi_path, roiName)]
        print(rois)
        # rois = [os.path.join(roi_path, 'PCa.nii')] + \
        #        [os.path.join(roi_path, 'BPH.nii')] + \
        #        [os.path.join(roi_path, 'SBPH.nii')] + \
        #        [os.path.join(roi_path, 'BPZ.nii')]

        # different brain tumor pathologies
        # rois = [os.path.join(roi_path, 't.nii.gz')] + \
        #        [os.path.join(roi_path, 'n.nii.gz')] + \
        #        [os.path.join(roi_path, 'h.nii.gz')] + \
        #        [os.path.join(roi_path, 'nc.nii.gz')] + \
        #        [os.path.join(roi_path, 'i.nii.gz')]

        # # different PCa Gleason scores
        # rois = [os.path.join(roi_path, 'G1.nii')] + \
        #        [os.path.join(roi_path, 'G2.nii')] + \
        #        [os.path.join(roi_path, 'G3.nii')] + \
        #        [os.path.join(roi_path, 'G4.nii')] + \
        #        [os.path.join(roi_path, 'G5.nii')]

        for roi in rois:
            stat = []
            try:
                atlas = nib.load(roi).get_data()
            except:
                print('No roi')
                continue

            roiFolder, roiName = os.path.split(roi)
            current_dir = dir
            roi_id = np.unique(atlas[atlas > 0])[0]
            idx = np.asarray(np.where(atlas == roi_id))
            print(len(param_list))
            for item in range(len(param_list)):
                print(param_list[item])
                print(glob.glob(os.path.join(current_dir, param_list[item]+'.nii')))
                img = nib.load(glob.glob(os.path.join(current_dir, param_list[item]+'.nii'))[0]).get_data()
                sub_data = img[atlas == roi_id]
                stat.append(sub_data)

            val = np.asarray(stat)
            # -4 for .nii file, -7 for nii.gz file
            val = np.concatenate((np.repeat(roiName[:-7], len(sub_data))[np.newaxis], val), axis=0)
            val = np.concatenate((np.repeat(roi_id, len(sub_data))[np.newaxis], val), axis=0)
            val = np.concatenate((np.asarray(range(0, len(sub_data)))[np.newaxis], val), axis=0)
            val = np.concatenate((idx, val), axis=0)
            val = np.concatenate((np.repeat(sub_id, len(sub_data))[np.newaxis], val), axis=0)
            val = np.transpose(val)

            df = pd.DataFrame(val, columns=col_list)
            stat_df = pd.concat([stat_df, df])

    stat_df.to_csv(os.path.join(saveFolder, caseName + 'NEW.csv'), index=False)
    print("extract voxels from ROI and DBSI: complete!!!")

def makeXPosFiles(dataFolder, saveFolder, x_inputs):
    allData = []
    for dirName, subdirList, fileList in os.walk(dataFolder):
        for i in range(len(fileList)):
            if fileList[i][0:3].isnumeric():
                patData = pd.read_csv(os.path.join(dataFolder,fileList[i]))
                # patData.fillna(0, inplace=True)
                for i in patData.columns[patData.isnull().any(axis=0)]:
                    # print(i)
                    patData[i].fillna(0, inplace=True)
                allData.append(patData)

    allDF = pd.concat(allData)
    # print(allDF.columns)
    allDF.columns = allDF.columns.str.replace(' ', '')
    dataX = allDF.iloc[:, x_input]
    # dataY = allDF['ROI_Class'].astype('int')
    dataPos = allDF.loc[:,['Sub_ID','X','Y','Z']]

    dataX.to_csv(os.path.join(saveFolder, 'excludeXNew.csv'))
    # dataY.to_csv(os.path.join(dataFolder, 'excludeYNew.csv'))
    dataPos.to_csv(os.path.join(saveFolder, 'excludePositionsNew.csv'))

dataFolder = r'Y:\Zezhong_Ye\Prostate_Cancer_Project_Shanghai\ALL_DBSI_DATA_new_thresh_all\PCa_with_biopsy'
saveFolder = r'C:\Users\atwu\Desktop\PCa_voxel_data\pro_data\excludeCSVONLY'
saveFolder2 = r'C:\Users\atwu\Desktop\PCa_voxel_data\pro_data'
caseName = r'001_ZHOU_CHAO_GANG'
roiName = r'PCa.nii'
thresholds = [0.1, 0.8, 2.3]
x_input = range(7, 30)
# extractVoxelFunc(dataFolder, saveFolder, caseName, roiName, thresholds)
makeXPosFiles(saveFolder, saveFolder2, x_input)