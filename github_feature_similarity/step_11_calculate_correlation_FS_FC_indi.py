# calculate the correlation of FC matrix across tasks for each participant
# calculate the correlation of FS matrix across tasks for each participant
# check whether the correlation of FS is greater than FS

import os
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from function_calculate_correlation_symmetric_matrix import calculate_correlation_symmetric_matrix_v2
from function_compare_correlation import  independent_corr
import nibabel as nib
from function_check_files_exist import check_files_exist
from scipy.stats import ttest_rel
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from function_plot_violin import star_sig
from scipy.io import loadmat
from function_calculate_correlation_symmetric_matrix import make_matrix_symmetric

# type = 'semantic'
type = 'non-semantic'

path_FS_sem = '/home/xiuyi/Data/Task_Gradient/Nifti/fMRIPrep_cifti_nifti_xcp_abcd/smooth_6_filter_min_0.01_0.08_36P_cifti/xcp_abcd_hctsa_double'
path_FS_nonsem = '/home/xiuyi/Data/Task_Gradient_localizers/Brain_data/Nifti/fMRIPrep_cifti_nifti_xcp_abcd/smooth_6_filter_min_0.01_0.08_36P_cifti/xcp_abcd_hctsa_double'

path_FC_sem    = '/home/xiuyi/Data/Task_Gradient/Nifti/fMRIPrep_cifti_nifti_xcp_abcd/smooth_6_filter_min_0.01_0.08_36P_cifti/ptseries_individual_parcellation_32k_task_regression_residual_FC_temporal_ac_xDF_indi'
path_FC_nonsem = '/home/xiuyi/Data/Task_Gradient_localizers/Brain_data/Nifti/fMRIPrep_cifti_nifti_xcp_abcd/smooth_6_filter_min_0.01_0.08_36P_cifti/ptseries_individual_parcellation_32k_task_regression_residual_FC_temporal_ac_xDF_indi'

path_PID_sem = '/home/xiuyi/Data/Task_Gradient/Nifti/fMRIPrep_cifti_nifti_xcp_abcd/smooth_6_filter_min_0.01_0.08_36P_cifti/ptseries_individual_parcellation_32k_synergy_redundancy'
path_PID_nonsem = '/home/xiuyi/Data/Task_Gradient_localizers/Brain_data/Nifti/fMRIPrep_cifti_nifti_xcp_abcd/smooth_6_filter_min_0.01_0.08_36P_cifti/ptseries_individual_parcellation_32k_synergy_redundancy'

suffix_FC = 'demean_parcel_merge_FC_z.pconn.nii'
suffix_FS = 'space-fsLR_den-91k_desc-residual_smooth_bold_demean_parcel_HCTSA_N_TSN_pearson_r_z.xlsx'
suffix_red = 'space-fsLR_den-91k_desc-residual_smooth_bold_red.mat'
suffix_syn = 'space-fsLR_den-91k_desc-residual_smooth_bold_syn.mat'

key_red = 'redundancy_mat'
key_syn = 'synergy_mat'

tasks_sem    = ['FeatureMatching','Association']
ses_IDs_sem  = ['ses-01','ses-02']

tasks_nonsem   = ['Spatial','Maths']
ses_IDs_nonsem = ['ses-02','ses-02']

path_output = '/home/xiuyi/Data/York_task_compare/FS_vs_FC_correlation/%s'%(type)

os.makedirs(path_output,exist_ok=True)

if type == 'semantic':
    path_FS = path_FS_sem
    path_FC = path_FC_sem
    path_PID = path_PID_sem
    tasks = tasks_sem
    ses_IDs = ses_IDs_sem
    title = 'Semantic: Feature Matching & Association'

elif type == 'non-semantic':
    path_FS = path_FS_nonsem
    path_FC = path_FC_nonsem
    path_PID = path_PID_nonsem
    tasks   = tasks_nonsem
    ses_IDs = ses_IDs_nonsem
    title = 'Non-Semantic: Spatial & Math'
sub_IDs = os.listdir(path_FS)

sub_IDs.sort()

folder_func = 'func'

r_FS_all = []
z_FS_all = []
p_FS_all = []

r_FC_all = []
z_FC_all = []
p_FC_all = []

r_red_all = []
z_red_all = []
p_red_all = []

r_syn_all = []
z_syn_all = []
p_syn_all = []

diff_FC_vs_FS_z_all = []
diff_FC_vs_FS_p_all = []

diff_red_vs_syn_z_all = []
diff_red_vs_syn_p_all = []

sub_IDs_valid = []
for sub_ID in sub_IDs:

    file_FS_1 = os.path.join(path_FS, sub_ID, ses_IDs[0], folder_func, '%s_%s_task-%s_%s'%( sub_ID,ses_IDs[0],tasks[0],suffix_FS))
    file_FS_2 = os.path.join(path_FS, sub_ID, ses_IDs[1], folder_func, '%s_%s_task-%s_%s'%( sub_ID,ses_IDs[1],tasks[1],suffix_FS))

    file_FC_1 = os.path.join(path_FC, sub_ID, ses_IDs[0], folder_func, '%s_%s_task-%s_%s'%( sub_ID,ses_IDs[0],tasks[0],suffix_FC))
    file_FC_2 = os.path.join(path_FC, sub_ID, ses_IDs[1], folder_func, '%s_%s_task-%s_%s'%( sub_ID,ses_IDs[1],tasks[1],suffix_FC))

    file_red_1 = os.path.join(path_PID, sub_ID, ses_IDs[0], folder_func, '%s_%s_task-%s_%s'%( sub_ID,ses_IDs[0],tasks[0],suffix_red))
    file_red_2 = os.path.join(path_PID, sub_ID, ses_IDs[1], folder_func, '%s_%s_task-%s_%s'%( sub_ID,ses_IDs[1],tasks[1],suffix_red))

    file_syn_1 = os.path.join(path_PID, sub_ID, ses_IDs[0], folder_func, '%s_%s_task-%s_%s'%( sub_ID,ses_IDs[0],tasks[0],suffix_syn))
    file_syn_2 = os.path.join(path_PID, sub_ID, ses_IDs[1], folder_func, '%s_%s_task-%s_%s'%( sub_ID,ses_IDs[1],tasks[1],suffix_syn))

    files_all = [file_FS_1, file_FS_2, file_FC_1, file_FC_2, file_red_1, file_red_2,  file_syn_1,  file_syn_2]

    if check_files_exist(files_all):
        sub_IDs_valid.append(sub_ID)
        data_FS_1 = np.array(pd.read_excel(file_FS_1).values)
        data_FS_2 = np.array(pd.read_excel(file_FS_2).values)

        data_FC_1 = nib.load(file_FC_1).get_fdata()
        data_FC_2 = nib.load(file_FC_2).get_fdata()

        data_red_1 = loadmat(file_red_1)[key_red]
        data_red_2 = loadmat(file_red_2)[key_red]

        data_syn_1 = loadmat(file_syn_1)[key_syn]
        data_syn_2 = loadmat(file_syn_2)[key_syn]

        data_red_1 = make_matrix_symmetric(data_red_1)
        data_red_2 = make_matrix_symmetric(data_red_2)

        data_syn_1 = make_matrix_symmetric(data_syn_1)
        data_syn_2 = make_matrix_symmetric(data_syn_2)

        r_FS, p_FS, num = calculate_correlation_symmetric_matrix_v2(data_FS_1,data_FS_2,method='spearman')
        z_FS = np.arctanh(r_FS)
        r_FS_all.append(r_FS)
        z_FS_all.append(z_FS)
        p_FS_all.append(p_FS)

        r_FC, p_FC, num = calculate_correlation_symmetric_matrix_v2(data_FC_1, data_FC_2,method='spearman')
        z_FC = np.arctanh(r_FC)
        r_FC_all.append(r_FC)
        z_FC_all.append(z_FC)
        p_FC_all.append(p_FC)

        r_red, p_red, num = calculate_correlation_symmetric_matrix_v2(data_red_1, data_red_2, method='spearman')
        z_red = np.arctanh(r_red)
        r_red_all.append(r_red)
        z_red_all.append(z_red)
        p_red_all.append(p_red)

        r_syn, p_syn, num = calculate_correlation_symmetric_matrix_v2(data_syn_1, data_syn_2, method='spearman')
        z_syn = np.arctanh(r_syn)
        r_syn_all.append(r_syn)
        z_syn_all.append(z_syn)
        p_syn_all.append(p_syn)

        diff_FC_vs_FS_z, diff_FC_vs_FS_p = independent_corr(r_FC, r_FS, n=num)

        diff_FC_vs_FS_z_all.append(diff_FC_vs_FS_z)
        diff_FC_vs_FS_p_all.append(diff_FC_vs_FS_p)

        diff_red_vs_syn_z, diff_red_vs_syn_p = independent_corr(r_red, r_syn, n=num)

        diff_red_vs_syn_z_all.append(diff_red_vs_syn_z)
        diff_red_vs_syn_p_all.append(diff_red_vs_syn_p)

df_stat = pd.DataFrame()

df_stat['participant'] = sub_IDs_valid
df_stat['FS_r'] = r_FS_all
df_stat['FS_z'] = z_FS_all
df_stat['FS_p'] = p_FS_all

df_stat['FC_r'] = r_FC_all
df_stat['FC_z'] = z_FC_all
df_stat['FC_p'] = p_FC_all

df_stat['red_r'] = r_red_all
df_stat['red_z'] = z_red_all
df_stat['red_p'] = p_red_all

df_stat['syn_r'] = r_syn_all
df_stat['syn_z'] = z_syn_all
df_stat['syn_p'] = p_syn_all

df_stat['FC_vs_FS_z'] =  diff_FC_vs_FS_z_all
df_stat['FC_vs_FS_p'] =  diff_FC_vs_FS_p_all

df_stat['red_vs_syn_z'] =  diff_red_vs_syn_z_all
df_stat['red_vs_syn_p'] =  diff_red_vs_syn_p_all

file_stat = os.path.join(path_output,'%s_FS_FC_PID.xlsx'%(type))
df_stat.to_excel(file_stat,index=False)

print ('well done')
# file_stat_long = os.path.join(path_output,'%s_FS_vs_FC_long.xlsx'%(type))
# fig_stat = os.path.join(path_output,'%s_FS_vs_FC.png'%(type))


#
# df_stat_part = df_stat[['participant','FS_z', 'FC_z']]
#
# df_stat_part_long = pd.melt(df_stat_part, id_vars=['participant'], value_vars=['z_FS', 'z_FC'], var_name='corr_type',
#                           value_name='corr_z')
#
# df_stat_part_long.to_excel(file_stat_long,index=False)
# t_ttest, p_ttest = ttest_rel( z_FC_all, z_FS_all)
# star_sig ='***'
# # plot the figures
# font_file = '/home/xiuyi/Data/Atlas/Playfair_Display/arial/arial.ttf'
# fontsize_label = 10
# sns.set(font_scale=2)
# sns.set_style("white")
# fig_size = (12, 12)
# fontsize_title = 25
# a4_dims = fig_size
# fig, ax = plt.subplots(figsize=a4_dims)
# order = ['z_FC','z_FS']
# sns.barplot(data=df_stat_part_long, x="corr_type", y="corr_z",palette=['#045275','#089099'], order = order)
# sns.swarmplot(ax=ax, x="corr_type", y="corr_z", data=df_stat_part_long, order=order, palette=['#003147','#003147'])
# fontprob_label=fm.FontProperties(fname=font_file, size=fontsize_label)
# ax.set_xticklabels(['FC','FS'],fontproperties=fontprob_label,fontsize=fontsize_title)
# ax.set_xlabel('', fontproperties=fontprob_label, fontsize=fontsize_title)
# ax.set_ylabel('correlation z', fontproperties=fontprob_label, fontsize=fontsize_title)
# ax.set_title(title, fontsize=fontsize_title,fontproperties=fontprob_label)
# ax.text(0.5, 0.5, star_sig, ha='center', va='bottom', color='#000000', fontproperties=fontprob_label)
#
# for i in range(len(z_FS_all)):
#     data_each = [z_FC_all[i],z_FS_all[i]]
#     sns.lineplot(data=data_each,markers=True,color='#808080')
#
#
# plt.savefig(fig_stat, dpi=300)
# plt.close()

print ('well done')
