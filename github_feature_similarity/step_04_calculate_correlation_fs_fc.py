# calculate the correlation between FC matrix and FS matrix
import numpy as np
from scipy.stats import pearsonr
from scipy.stats import spearmanr
import pandas as pd
from function_calculate_correlation_symmetric_matrix import calculate_correlation_symmetric_matrix
import os

file_1 = '/home/xiuyi/Data/HCP/11_FC_xDF_z_mean_group/HCP_rest_FC_xDF.xlsx'
file_2 = '/home/xiuyi/Data/HCP/99_Results/hctsa_timeseries_fs/rest_fs_similarity_z.xlsx'

path_sem_FC = '/home/xiuyi/Data/Task_Gradient/Nifti/fMRIPrep_cifti_nifti_xcp_abcd/smooth_6_filter_min_0.01_0.08_36P_cifti/ptseries_individual_parcellation_32k_task_regression_residual_FC_temporal_ac_xDF_group'
path_sem_FS = '/home/xiuyi/Data/Task_Gradient/Nifti/fMRIPrep_cifti_nifti_xcp_abcd/smooth_6_filter_min_0.01_0.08_36P_cifti/xcp_abcd_hctsa_double_tsn/pearson_r_z'

path_non_sem_FC = '/home/xiuyi/Data/Task_Gradient_localizers/Brain_data/Nifti/fMRIPrep_cifti_nifti_xcp_abcd/smooth_6_filter_min_0.01_0.08_36P_cifti/ptseries_individual_parcellation_32k_task_regression_residual_FC_temporal_ac_xDF_group'
path_non_sem_FS = '/home/xiuyi/Data/Task_Gradient_localizers/Brain_data/Nifti/fMRIPrep_cifti_nifti_xcp_abcd/smooth_6_filter_min_0.01_0.08_36P_cifti/xcp_abcd_hctsa_double_tsn/pearson_r_z'
file_3 = os.path.join(path_sem_FC,'ses-01_task-FeatureMatching_demean_parcel_merge_FC_z_mean.xlsx')
file_4 = os.path.join(path_sem_FS,'ses-01_task-FeatureMatching_space-fsLR_den-91k_desc-residual_smooth_bold_demean_parcel_HCTSA_N_TSN_pearson_r_z.xlsx')

file_5 = os.path.join(path_sem_FC,'ses-02_task-Association_demean_parcel_merge_FC_z_mean.xlsx')
file_6 = os.path.join(path_sem_FS,'ses-02_task-Association_space-fsLR_den-91k_desc-residual_smooth_bold_demean_parcel_HCTSA_N_TSN_pearson_r_z.xlsx')

file_7 = os.path.join(path_non_sem_FC,'ses-02_task-Spatial_demean_parcel_merge_FC_z_mean.xlsx')
file_8 = os.path.join(path_non_sem_FS,'ses-02_task-Spatial_space-fsLR_den-91k_desc-residual_smooth_bold_demean_parcel_HCTSA_N_TSN_pearson_r_z.xlsx')


file_9 = os.path.join(path_non_sem_FC,'ses-02_task-Maths_demean_parcel_merge_FC_z_mean.xlsx')
file_10 = os.path.join(path_non_sem_FS,'ses-02_task-Maths_space-fsLR_den-91k_desc-residual_smooth_bold_demean_parcel_HCTSA_N_TSN_pearson_r_z.xlsx')

perm_num = 10000
tasks = ['rest','FM','Asso','Spatial','Maths']
files_FC = [file_1,file_3,file_5,file_7,file_9]
files_FS = [file_2,file_4,file_6,file_8,file_10]

for i in range(len(files_FC)):

    print (tasks[i])

    mat_1 = pd.read_excel(files_FC[i]).values
    mat_2 = pd.read_excel(files_FS[i]).values

    r_real,p_real,p_perm_all, count, perm_num = calculate_correlation_symmetric_matrix(mat_1,mat_2,perm_num,method='pearsonr')
