# run the gradient analysis at the group level of the HCP rest FS matrix

import os
import nibabel as nib
import numpy as np
import pandas as pd
from  scipy.stats import pearsonr
from brainspace.gradient import GradientMaps
import warnings
warnings.simplefilter('ignore')
from os.path import  join
from scipy.stats import spearmanr
from scipy.io import loadmat
from function_plot_Kong_ptseries_dlabel import plot_Kong_parcellation_2
from function_edit_scene_figures import add_colorbar2
import shutil
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

parcel_num = 400

path_base_group = '/home/xiuyi/Data/HCP/11_FC_xDF_z_mean_group'

# cifti template for saving new data
file_template = '/home/xiuyi/Data/Atlas/cifti_template/Schaefer_400_17_group.ptseries.nii'
cifti_template = nib.load(file_template)

# parameters related to gradients analysis
# set number of components
comp_num = 8

# try different parameters for gradients analysis
kernels = [None,'pearson', 'spearman', 'normalized_angle']
embeddings = ['pca', 'le', 'dm']
alignments = ['joint', 'procrustes']


kernel = kernels[0]
embedding = embeddings[2]
alignment = alignments[0]
task ='HCP_rest'
# for kernel in kernels:
#     for embedding in embeddings:
#         for alignment in alignments:
path_output_group = '%s/kernel_%s_embedding_%s'%(path_base_group ,kernel, embedding)
os.makedirs(path_output_group,exist_ok=True)

file_fc_group = '/home/xiuyi/Data/HCP/11_FC_xDF_z_mean_group/HCP_rest_FC_xDF.xlsx'

fc_matrix_group = pd.read_excel(file_fc_group)

# run the gradient analysis for the group average fc
# including alignment
gradient_setting_group = GradientMaps(n_components=comp_num, approach=embedding, kernel=kernel)

# not including alignment
# gradient_setting_group = GradientMaps(n_components=comp_num, approach=embedding, kernel=kernel)
gradient_group = gradient_setting_group.fit(fc_matrix_group.values)


fig_scree = join(path_output_group, '%s_FC_gradient_scree_kernel-%s_embedding-%s.png' % (task,  kernel, embedding))
fig, ax = plt.subplots(1, figsize=(5, 4))
ax.scatter(range(1,gradient_setting_group.lambdas_.size+1), gradient_setting_group.lambdas_)
ax.set_title('%s_FC_grad_scree_kernel-%s_embedding-%s' % (task,  kernel, embedding))
ax.set_xlabel('Component Nb')
ax.set_ylabel('Eigenvalue')

plt.savefig(fig_scree,dpi=300)
plt.close()

print (gradient_setting_group.lambdas_[0])
print (gradient_setting_group.lambdas_[1])
print (gradient_setting_group.lambdas_[2])

print ('total variance: ',gradient_setting_group.lambdas_[0] + gradient_setting_group.lambdas_[1] + gradient_setting_group.lambdas_[2] )

# save each gradient as *ptseries.nii
for comp_id in range(comp_num):

    grad_ids = gradient_group.gradients_[:, comp_id]
    # save each gradient
    file_map = join(path_output_group, '%s_FC_gradient_%s_kernel-%s_embedding-%s.ptseries.nii' % (task, comp_id + 1, kernel, embedding))
    file_map_flip = join(path_output_group, '%s_FC_gradient_%s_kernel-%s_embedding-%s_flip.ptseries.nii' % (task, comp_id + 1, kernel, embedding))
    fig = join(path_output_group, '%s_FC_gradient_%s_kernel-%s_embedding-%s.png' % (task, comp_id + 1, kernel, embedding))
    fig_flip = join(path_output_group, '%s_FC_gradient_%s_kernel-%s_embedding-%s_flip.png' % (task, comp_id + 1, kernel, embedding))
    my_values = grad_ids.reshape(1, parcel_num)
    if not os.path.exists(file_map):
        new_img = nib.Cifti2Image(my_values, header=cifti_template.header, nifti_header=cifti_template.nifti_header)
        new_img.to_filename(file_map)
    if not os.path.exists(fig):
    # if  os.path.exists(fig):
        title = '%s_FC_gradient_%s_kernel-%s_embedding-%s' % (task, comp_id + 1, kernel, embedding)
        # plot_Kong_parcellation_2(my_values[0], fig, 'nipy_spectral', title, title_position=100)
        plot_Kong_parcellation_2(my_values[0], fig, 'jet', title, title_position=100)

    my_values_flip = (my_values * (-1)).reshape(1, parcel_num)

    if not os.path.exists(file_map_flip):
        new_img = nib.Cifti2Image(my_values_flip, header=cifti_template.header, nifti_header=cifti_template.nifti_header)
        new_img.to_filename(file_map_flip)

    if not os.path.exists(fig_flip):
        title = '%s_FC_gradient_%s_kernel-%s_embedding-%s_flip' % (task, comp_id + 1, kernel, embedding)
        # plot_Kong_parcellation_2(my_values_flip[0], fig_flip, 'nipy_spectral',title, title_position=100)
        plot_Kong_parcellation_2(my_values_flip[0], fig_flip, 'jet',title, title_position=100)

print ('well done group gradient')


