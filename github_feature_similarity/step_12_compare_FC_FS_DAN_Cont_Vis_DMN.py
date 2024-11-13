# convert FS (400 * 400) to (12 * 12)
import os
import pandas as pd
import numpy as np
from function_convert_matrix_412_to_12 import load_matrix_file, convert_FC_412_networks
from scipy.io import loadmat,savemat
import nibabel as nib

save_SC = 2
save_FC_rest = 2
save_FC_task = 2

save_FS_rest = 2
save_FS_task = 1

networks = ['01_Vis',   '02_Aud',    '03_SM',     '04_VAN-A',  '05_VAN-B', '06_DAN-A',
            '07_DAN-B', '08_Cont-A', '09_Cont-B', '10_Cont-C', '11_Lang',  '12_DMN']

key = 'data_network'
path_output = '/home/xiuyi/Data/York_task_compare/FC_vs_FC_specific_network_pairs/data_network'
os.makedirs(path_output,exist_ok=True)

path_FS_rest = '/home/xiuyi/Data/HCP/12_HCP_hctsa_feature_similarity'
path_FS_sem = '/home/xiuyi/Data/Task_Gradient/Nifti/fMRIPrep_cifti_nifti_xcp_abcd/smooth_6_filter_min_0.01_0.08_36P_cifti/xcp_abcd_hctsa_double'
path_FS_nonsem = '/home/xiuyi/Data/Task_Gradient_localizers/Brain_data/Nifti/fMRIPrep_cifti_nifti_xcp_abcd/smooth_6_filter_min_0.01_0.08_36P_cifti/xcp_abcd_hctsa_double'

path_FC_rest = '/home/xiuyi/Data/HCP/11_FC_xDF_z_mean'
path_FC_sem    = '/home/xiuyi/Data/Task_Gradient/Nifti/fMRIPrep_cifti_nifti_xcp_abcd/smooth_6_filter_min_0.01_0.08_36P_cifti/ptseries_individual_parcellation_32k_task_regression_residual_FC_temporal_ac_xDF_indi'
path_FC_nonsem = '/home/xiuyi/Data/Task_Gradient_localizers/Brain_data/Nifti/fMRIPrep_cifti_nifti_xcp_abcd/smooth_6_filter_min_0.01_0.08_36P_cifti/ptseries_individual_parcellation_32k_task_regression_residual_FC_temporal_ac_xDF_indi'


#%%
if save_FS_rest == 1:
    file_12 = os.path.join(path_output, '06_FS_HCPA-rest.mat')
    sub_IDs = os.listdir(path_FS_rest)
    sub_IDs.sort()
    network_indi = np.zeros((len(sub_IDs), 12, 12))
    for i in range(len(sub_IDs)):
 
        sub_ID = sub_IDs[i]
        data_indi_all = np.zeros((4,400,400))
        for run_ID in range(1,5):
            file = os.path.join(path_FS_rest,sub_ID,'%s_task-rest_run-%s_demean_parcel_HCTSA_N_fs.mat'%(sub_ID,run_ID))
            data_indi_each = loadmat(file)['similarity_z']
            data_indi_all[run_ID-1]=data_indi_each

        # calculate the mean for each subject
        data_indi = np.mean(data_indi_all,axis=0)
        # convert it to 12 * 12
        df_indi = load_matrix_file(data_indi)
        df_12 =  convert_FC_412_networks(df_indi)

        # save it to one mat: N * 12 * 12
        network_indi[i,:,:] =df_12.values

    # save it
    network_indi_key = {key:network_indi, 'network':networks}
    savemat(file_12,network_indi_key)
    data_test = loadmat(file_12)


#%% save the FS of tasks
if save_FS_task == 1:
    file_suffix = 'space-fsLR_den-91k_desc-residual_smooth_bold_demean_parcel_HCTSA_N_TSN_pearson_r_z.xlsx'

    for type in ['sem','nonsem']:
        if type == 'sem':
            path_FS = path_FS_sem
            tasks = ['FeatureMatching','Association']
            ses_IDs = ['ses-01','ses-02']
            mat_IDs = ['09', '10']

        elif type == 'nonsem':
            path_FS = path_FS_nonsem
            tasks   = ['Spatial', 'Maths']
            ses_IDs = ['ses-02', 'ses-02']
            mat_IDs = ['07', '08']

        sub_IDs = os.listdir(path_FS)
        sub_IDs.sort()

        for m in range(len(tasks)):
            task = tasks[m]
            ses_ID = ses_IDs[m]
            mat_ID = mat_IDs[m]

            file_12 = os.path.join(path_output, '%s_FS_%s.mat'%(mat_ID,task))
            network_indi = np.zeros((len(sub_IDs), 12, 12))
            i_valid = 0
            for i in range(len(sub_IDs)):

 
                sub_ID = sub_IDs[i]
                file_input = os.path.join(path_FS,sub_ID,ses_ID,'func','%s_%s_task-%s_%s'%(sub_ID,ses_ID,task,file_suffix))

                if os.path.exists(file_input):
                    data_indi = pd.read_excel(file_input)

                    # convert it to 12 * 12
                    df_indi = load_matrix_file(data_indi)
                    df_12 =  convert_FC_412_networks(df_indi)

                    # save it to one mat: N * 12 * 12
                    network_indi[i_valid,:,:] =df_12.values
                    i_valid +=1
                else:
                    print(task,sub_ID,'not exist')
            # only keep the valid data
            network_indi_valid = network_indi[0:i_valid-1,:,:]
            # save it
            network_indi_key = {key:network_indi_valid, 'network':networks}
            savemat(file_12,network_indi_key)
            data_test = loadmat(file_12)

#%% save the FC of tasks
if save_FC_task == 1:
    file_suffix = 'demean_parcel_merge_FC_z.pconn.nii'

    for type in ['sem','nonsem']:
        if type == 'sem':
            path_FC = path_FC_sem
            tasks = ['FeatureMatching','Association']
            ses_IDs = ['ses-01','ses-02']
            mat_IDs = ['03', '04']
        elif type == 'nonsem':
            path_FC = path_FC_nonsem
            tasks   = ['Spatial', 'Maths']
            ses_IDs = ['ses-02', 'ses-02']
            mat_IDs = ['01', '02']

        sub_IDs = os.listdir(path_FC)
        sub_IDs.sort()

        for m in range(len(tasks)):
            task = tasks[m]
            ses_ID = ses_IDs[m]
            mat_ID = mat_IDs[m]

            file_12 = os.path.join(path_output, '%s_FC_%s.mat'%(mat_ID,task))
            network_indi = np.zeros((len(sub_IDs), 12, 12))
            i_valid = 0
            for i in range(len(sub_IDs)):

            # for i in range(2):
                sub_ID = sub_IDs[i]
                file_input = os.path.join(path_FC,sub_ID,ses_ID,'func','%s_%s_task-%s_%s'%(sub_ID,ses_ID,task,file_suffix))

                if os.path.exists(file_input):
                    data_indi = np.array(nib.load(file_input).get_fdata())

                    # convert it to 12 * 12
                    df_indi = load_matrix_file(data_indi)
                    df_12 =  convert_FC_412_networks(df_indi)

                    # save it to one mat: N * 12 * 12
                    network_indi[i_valid,:,:] =df_12.values
                    i_valid +=1
                else:
                    print(task,sub_ID,'not exist')
            # only keep the valid data
            network_indi_valid = network_indi[0:i_valid-1,:,:]
            # save it
            network_indi_key = {key:network_indi_valid, 'network':networks}
            savemat(file_12,network_indi_key)
            data_test = loadmat(file_12)

#%% read the FC data of rest
if save_FC_rest == 1:
    file_12 = os.path.join(path_output, '00_FC_HCPA-rest.mat')
    sub_IDs = os.listdir(path_FC_rest)
    sub_IDs.sort()
    network_indi = np.zeros((len(sub_IDs), 12, 12))
    for i in range(len(sub_IDs)):
        sub_ID = sub_IDs[i]
        file = os.path.join(path_FC_rest,sub_ID)
        data_indi = loadmat(file)['z_corr_data']

        # convert it to 12 * 12
        df_indi = load_matrix_file(data_indi)
        df_12 =  convert_FC_412_networks(df_indi)

        # save it to one mat: N * 12 * 12
        network_indi[i,:,:] =df_12.values

    # save it
    network_indi_key = {key:network_indi, 'network':networks}
    savemat(file_12,network_indi_key)
    data_test = loadmat(file_12)

print ('well done')
