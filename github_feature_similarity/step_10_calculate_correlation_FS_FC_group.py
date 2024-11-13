# calculate the correlation of FC matrix across tasks
# calculate the correlation of FS matrix across tasks
# check whether the correlation of FS is greater than FS

import os
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from function_calculate_correlation_symmetric_matrix import calculate_correlation_symmetric_matrix_v2
from function_compare_correlation import  independent_corr
path_FS_sem = '/home/xiuyi/Data/Task_Gradient/Nifti/fMRIPrep_cifti_nifti_xcp_abcd/smooth_6_filter_min_0.01_0.08_36P_cifti/xcp_abcd_hctsa_double_tsn/pearson_r_z'
path_FS_nonsem = '/home/xiuyi/Data/Task_Gradient_localizers/Brain_data/Nifti/fMRIPrep_cifti_nifti_xcp_abcd/smooth_6_filter_min_0.01_0.08_36P_cifti/xcp_abcd_hctsa_double_tsn/pearson_r_z'
path_FS_rest = '/home/xiuyi/Data/HCP/99_Results/hctsa_timeseries_fs'

path_FC_sem = '/home/xiuyi/Data/Task_Gradient/Nifti/fMRIPrep_cifti_nifti_xcp_abcd/smooth_6_filter_min_0.01_0.08_36P_cifti/ptseries_individual_parcellation_32k_task_regression_residual_FC_temporal_ac_xDF_group'
path_FC_nonsem = '/home/xiuyi/Data/Task_Gradient_localizers/Brain_data/Nifti/fMRIPrep_cifti_nifti_xcp_abcd/smooth_6_filter_min_0.01_0.08_36P_cifti/ptseries_individual_parcellation_32k_task_regression_residual_FC_temporal_ac_xDF_group'
path_FC_rest = '/home/xiuyi/Data/HCP/11_FC_xDF_z_mean_group'

suffix_FC_rest = 'HCP_rest_FC_xDF.xlsx'
suffix_FC = 'demean_parcel_merge_FC_z_mean.xlsx'

suffix_FS_rest = 'rest_fs_similarity_z.xlsx'
suffix_FS = 'space-fsLR_den-91k_desc-residual_smooth_bold_demean_parcel_HCTSA_N_TSN_pearson_r_z.xlsx'

tasks = ['rest','FeatureMatching','Association','Spatial','Maths']
ses_IDs = ['no_session','ses-01','ses-02','ses-02','ses-02']

file_FS_rest = os.path.join(path_FS_rest,suffix_FS_rest)
file_FS_Feat = os.path.join(path_FS_sem,'%s_task-%s_%s'%( ses_IDs[1],tasks[1],suffix_FS))
file_FS_Asso = os.path.join(path_FS_sem,'%s_task-%s_%s'%( ses_IDs[2],tasks[2],suffix_FS))
file_FS_Spatial = os.path.join(path_FS_nonsem,'%s_task-%s_%s'%( ses_IDs[3],tasks[3],suffix_FS))
file_FS_Maths   = os.path.join(path_FS_nonsem,'%s_task-%s_%s'%( ses_IDs[4],tasks[4],suffix_FS))


file_FC_rest    = os.path.join(path_FC_rest,suffix_FC_rest)
file_FC_Feat    = os.path.join(path_FC_sem,'%s_task-%s_%s'%( ses_IDs[1],tasks[1],suffix_FC))
file_FC_Asso    = os.path.join(path_FC_sem,'%s_task-%s_%s'%( ses_IDs[2],tasks[2],suffix_FC))
file_FC_Spatial = os.path.join(path_FC_nonsem,'%s_task-%s_%s'%( ses_IDs[3],tasks[3],suffix_FC))
file_FC_Maths   = os.path.join(path_FC_nonsem,'%s_task-%s_%s'%( ses_IDs[4],tasks[4],suffix_FC))


files_FS = [file_FS_rest,file_FS_Feat,file_FS_Asso,file_FS_Spatial, file_FS_Maths]
files_FC = [file_FC_rest,file_FC_Feat,file_FC_Asso,file_FC_Spatial, file_FC_Maths]

data_FS_all = np.zeros((5,400,400))
data_FC_all = np.zeros((5,400,400))
i = 0
for file in files_FS:
    data = pd.read_excel(file)

    data_np = np.array(data.values)

    data_FS_all[i,:,:]=data_np

    i = i + 1

i = 0
for file in files_FC:
    data = pd.read_excel(file)
    if data.shape[0]< 400:
        data = pd.read_excel(file,header=None)

    data_np = np.array(data.values)

    data_FC_all[i,:,:]=data_np

    i = i + 1

r_FS_all = []
p_FS_all = []

r_FC_all = []
p_FC_all = []
pairs = []
for i in range(4):
    data_FS_1 = data_FS_all[i, :, :]

    for j in range(i+1, 5):

        data_FS_2 = data_FS_all[j,:,:]

        r_FS, p_FS, num = calculate_correlation_symmetric_matrix_v2(data_FS_1,data_FS_2,method='spearman')

        r_FS_all.append(r_FS)
        p_FS_all.append(p_FS)

        data_FC_1 = data_FC_all[i, :, :]
        data_FC_2 = data_FC_all[j, :, :]

        r_FC, p_FC, num = calculate_correlation_symmetric_matrix_v2(data_FC_1, data_FC_2,method='spearman')

        r_FC_all.append(r_FC)
        p_FC_all.append(p_FC)

        diff_z, diff_p = independent_corr(r_FC,r_FS,n=num,)

        print('                ')

        pairs.append('%s vs %s '% (tasks[i], tasks[j]))
        print (tasks[i],tasks[j])

        print (r_FS,p_FS)
        print (r_FC,p_FC)
        print (diff_z, diff_p)
        print ('                ')
        print ('                ')


print (pairs)
print (r_FS_all)
print (r_FC_all)