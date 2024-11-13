# read the normalized data of hctsa
# calculate the temporal profile similarity
# get a matrix
import os
import hdf5storage
from scipy.io import loadmat,savemat
import numpy as np
import pandas as pd
import mat73
import numpy as np
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine
from os.path import join
from function_convert_matrix_417_to_17 import  load_matrix_file, convert_FC_417_networks, plot_matrix_whole, plot_FC_matrix_network
from scipy.stats import ttest_rel
from matplotlib.colors import ListedColormap
from function_generate_contrast_pairs import generate_contrast_pairs_all
from scipy.stats import ttest_1samp
from itertools import combinations
method_similarity = 'pearson_r_z'
dataset = 2
if dataset == 1:
    path_output_base = '/home/xiuyi/Data/Task_Gradient/Nifti/fMRIPrep_cifti_nifti_xcp_abcd/smooth_6_filter_min_0.01_0.08_36P_cifti/xcp_abcd_hctsa_double'
    path_output_group = '/home/xiuyi/Data/Task_Gradient/Nifti/fMRIPrep_cifti_nifti_xcp_abcd/smooth_6_filter_min_0.01_0.08_36P_cifti/xcp_abcd_hctsa_double_tsn/%s'%(method_similarity)
    ses_IDs = ['ses-01', 'ses-02']
    tasks = ['FeatureMatching','Association']

elif dataset == 2:
    path_output_base = '/home/xiuyi/Data/Task_Gradient_localizers/Brain_data/Nifti/fMRIPrep_cifti_nifti_xcp_abcd/smooth_6_filter_min_0.01_0.08_36P_cifti/xcp_abcd_hctsa_double'
    path_output_group = '/home/xiuyi/Data/Task_Gradient_localizers/Brain_data/Nifti/fMRIPrep_cifti_nifti_xcp_abcd/smooth_6_filter_min_0.01_0.08_36P_cifti/xcp_abcd_hctsa_double_tsn/%s' % (method_similarity)
    # ses_IDs = ['ses-01','ses-02','ses-02', 'ses-02']
    # tasks = ['FeatureMatching','Language','Spatial', 'Maths']
    ses_IDs = ['ses-01','ses-02']
    tasks = ['FeatureMatching','Language']

folder_func = 'func'

cmap = ListedColormap(['darkblue', 'blue', 'cyan','green','lightgreen','yellow','orange','orangered','red','darkred'])


os.makedirs(path_output_group, exist_ok=True)

file_suffix_hctsa = 'space-fsLR_den-91k_desc-residual_smooth_bold_demean_parcel_HCTSA_N_TS_DataMat.mat'
file_suffix_xlsx = 'space-fsLR_den-91k_desc-residual_smooth_bold_demean_parcel_HCTSA_N_TSN_%s.xlsx'%(method_similarity)
file_suffix_mat = 'space-fsLR_den-91k_desc-residual_smooth_bold_demean_parcel_HCTSA_N_TSN_%s.mat'%(method_similarity)


sub_IDs = os.listdir(path_output_base)

sub_IDs.sort()

cols = [i for i in range(1,401)]
df_1samp = pd.DataFrame()
df = pd.DataFrame()
for i in range(len(ses_IDs)):
    ses_ID = ses_IDs[i]
    task = tasks[i]

    if task == 'FeatureMatching' or task == 'Association':
        run_num = 4
    else:
        run_num = 2
    for sub_ID in sub_IDs:
        print (sub_ID)

        # create a df to store the interested feature
        for run in range(1,run_num+1):
            print (run)
            filename_hctsa = '%s_%s_task-%s_run-%s_%s' % (sub_ID, ses_ID, task, run, file_suffix_hctsa)
            file_hctsa = join(path_output_base, sub_ID, ses_ID, folder_func, filename_hctsa)

            filename_xlsx = '%s_%s_task-%s_run-%s_%s' % (sub_ID, ses_ID, task, run, file_suffix_xlsx)
            file_xlsx = join(path_output_base, sub_ID, ses_ID, folder_func, filename_xlsx)

            filename_mat = '%s_%s_task-%s_run-%s_%s' % (sub_ID, ses_ID, task, run, file_suffix_mat)
            file_mat = join(path_output_base, sub_ID, ses_ID, folder_func, filename_mat)


            if not os.path.exists(file_xlsx) and not os.path.exists(file_mat):
                if not os.path.exists(file_hctsa):
                    print ('file_hctsa not exists %s' % (file_hctsa))

                else:
                    # data_dict = mat73.loadmat(file_hctsa)

                    # data_dict = hdf5storage.loadmat(file_hctsa)
                    # TS_DataMat = np.array(data_dict['TS_DataMat'])
                    data_dict = loadmat(file_hctsa)
                    TS_DataMat = np.array(data_dict['data'])

                    # calculate the pearson r similarity
                    dist_out_r = np.corrcoef(TS_DataMat)

                    # fill the diagonal with 0
                    np.fill_diagonal( dist_out_r, 0)
                    # convert r to z
                    dist_out = np.arctanh(dist_out_r)

                    # save it to excel
                    try:
                        # dist_out_key = {'FS_z',dist_out}
                        # savemat(dist_out_key,file_mat)

                        data = pd.DataFrame(data = dist_out,columns =cols)
                        data.to_excel(file_xlsx,index=False)
                    except:
                        print (' no save %s'%(file_xlsx))


print('done')

