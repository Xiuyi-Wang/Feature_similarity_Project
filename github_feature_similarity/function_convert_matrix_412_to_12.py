# convert the 400 * 400 network-based FC matrix to 12 * 12 network-based FC matrix or 34 * 34
# merge all the visual-A,B, C as visual
# merge all the DMN-A,B,C as DMN
# merge all the SM-A, B as SM
# the input is 400 * 400 matrix

import pandas as pd
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
# from netneurotools import plotting
from PIL import Image, ImageFont, ImageDraw



file_FC_parcel_400_1 = '/home/xiuyi/Data/Task_Gradient/Nifti/fMRIPrep_cifti_nifti_xcp_abcd/smooth_6_filter_min_0.01_0.08_36P_cifti/xcp_abcd_FC_gradient_timescale_group/FC/ses-01_task-FeatureMatching_demean_parcel_merge_FC_z.xlsx'

file_FC_parcel_400_2 = '/home/xiuyi/Data/Task_Gradient/Nifti/fMRIPrep_cifti_nifti_xcp_abcd/smooth_6_filter_min_0.01_0.08_36P_cifti/xcp_abcd_FC_gradient_timescale_group/ses-02_task-rest_demean_parcel_merge_FC_z.pconn.nii'

col_hemi = 'hemi'
col_network = 'network_ID_name'
file_atlas = '/home/xiuyi/Data/Atlas/Parcellation_Kong_2022/Schaefer_412.xlsx'
cmap = 'nipy_spectral'
font_file = '/home/xiuyi/Data/Atlas/Playfair_Display/static/PlayfairDisplay-Regular.ttf'
font = ImageFont.truetype(font=font_file,size=40)

def load_atlas(file_atlas):
    """
    :param file_atlas: is the Kong_2022_Schaefer_2018 file_hctsa
    :return:
    """
    data_atlas = pd.read_excel(file_atlas)

    return data_atlas

def load_matrix_file(file_FC_parcel_400, sort=True):
    """
       :param file_FC_parcel_400: is the 400 * 400 FC - z values; can be pconn.nii, xlsx, array
                                 the order is network 1, 2, 3,,,400
       :param network_num: can be 34, LH 17 and RH 17; 17: Bilateral
       :param sort: whether sort the column name and index name; true, sort; false no sort
       :return: 400 * 400, the order is LH vis 1,2,3 .. DMN 1, 2, 3; RH Vis 1, 2, 3 .. DMN 1, 2, 3
       """

    if type(file_FC_parcel_400) == np.ndarray:
        # in this case, the file_stat_parcel_400 is an array, the shape is 400 * 400
        data_FC_parcel = pd.DataFrame(data=file_FC_parcel_400)

    elif type(file_FC_parcel_400) == pd.core.frame.DataFrame:
        data_FC_parcel = file_FC_parcel_400

    elif file_FC_parcel_400.endswith('.xlsx'):
        data_FC_parcel = pd.read_excel(file_FC_parcel_400, header=None)

    elif file_FC_parcel_400.endswith('.pconn.nii'):
        # load the file_hctsa, get the data and convert it to df
        data_FC_parcel = pd.DataFrame(data=nib.load(file_FC_parcel_400).get_fdata())

    # load the data_atlas
    data_atlas = load_atlas(file_atlas)

    col_add = col_network

    column_names = list(data_atlas[col_add].values)


    # add the column names
    data_FC_parcel.columns = column_names

    # add one column
    data_FC_parcel[col_network] = column_names

    # set it as index
    data_FC_parcel.set_index(col_network, inplace=True)

    # sort it by column names and index
    if sort:
        data_FC_parcel.sort_index(inplace=True)
        data_FC_parcel.sort_index(axis=1, inplace=True)

    return data_FC_parcel



def convert_FC_412_networks(data_FC_parcel,corr_value = 'z'):
    """
    :param file_parcel_400: is the 400 * 400 FC - z values; can be pconn.nii, xlsx, array
                            the order is network 1, 2, 3,,,400
    :return: array_12*12
    """

    # transform r to z
    if corr_value == 'r':
        np.fill_diagonal(data_FC_parcel.values, 0)
        data_FC_parcel = pd.DataFrame(data= np.arctanh(data_FC_parcel.values),columns = data_FC_parcel.columns.values, index=data_FC_parcel.index)

    elif corr_value == 'z':
        # replace the diagonal with zero
        np.fill_diagonal(data_FC_parcel.values, 0)

    # convert index to a column
    data_FC_parcel_1 = data_FC_parcel.reset_index(level=0)

    # calculate the mean of each network
    data_FC_parcel_mean = data_FC_parcel_1.groupby(col_network).mean()

    # transpose the data_FC_parcel_for_34
    data_FC_parcel_mean_T = data_FC_parcel_mean.transpose()

    # reset the index as a column
    data_FC_parcel_mean_T = data_FC_parcel_mean_T.reset_index()

    # calculate the mean of each network again
    data_FC_parcel_mean_T_mean = data_FC_parcel_mean_T.groupby('index').mean()

    return data_FC_parcel_mean_T_mean



def load_matrix_file_half(file_mat_200, hemi, sort=True):
    """
       :param file_FC_parcel_400: is the 400 * 400 FC - z values; can be pconn.nii, xlsx, array
                                 the order is network 1, 2, 3,,,400
       :param network_num: can be 34, LH 17 and RH 17; 17: Bilateral
       :param sort: whether sort the column name and index name; true, sort; false no sort
       :return: 400 * 400, the order is LH vis 1,2,3 .. DMN 1, 2, 3; RH Vis 1, 2, 3 .. DMN 1, 2, 3
       """

    if type(file_mat_200) == np.ndarray:
        # in this case, the file_stat_parcel_400 is an array, the shape is 400 * 400
        data_mat_200 = pd.DataFrame(data=file_mat_200)

    elif type(file_mat_200) == pd.core.frame.DataFrame:
        data_mat_200 = file_mat_200

    elif file_mat_200.endswith('.xlsx'):
        data_mat_200 = pd.read_excel(file_mat_200, header=None)

    elif file_mat_200.endswith('.pconn.nii'):
        # load the file_hctsa, get the data and convert it to df
        data_mat_200 = pd.DataFrame(data=nib.load(file_mat_200).get_fdata())

    # load the data_atlas
    data_atlas = load_atlas(file_atlas)

    col_add = col_network

    column_names_all = list(data_atlas[col_add].values)
    if hemi =='L':
        column_names = column_names_all[0:200]
    elif hemi =='R':
        column_names = column_names_all[200::]
    # add the column names
    data_mat_200.columns = column_names

    # add one column
    data_mat_200[col_network] = column_names

    # set it as index
    data_mat_200.set_index(col_network, inplace=True)

    # sort it by column names and index
    if sort:
        data_mat_200.sort_index(inplace=True)
        data_mat_200.sort_index(axis=1, inplace=True)

    return data_mat_200






def convert_mat_412_N_networks(matrix,network_num = 34):
    """
    :param matrix: is 400 * N matrix; example: the autocorrelation of networks in N lags
    :param network_num: 34: LH + RH or 17
    :return: matrix_network: shape: network_num * N
    """
    # create a dataframe, add the column names
    # load the data_atlas
    data_atlas = load_atlas(file_atlas)

    if network_num == 34:
        column_names = list(data_atlas['hemi_network_ID_17'].values)
    elif network_num == 17:
        column_names = list(data_atlas['network_ID_name'].values)

    data_mat = pd.DataFrame(data = matrix)

    # add one column
    data_mat[col_hemi_network] = column_names

    # calculate the mean of each network
    data_mat_mean =  data_mat.groupby(col_hemi_network).mean()

    # add network names

    return data_mat_mean




def plot_matrix_whole(data_FC_parcel, file_fig,title, title_position=300, corr_value = 'r'):
    """
    :param data_FC_parcel: is 400 * 400 matrix, sorted by hemi_network
    :return: a fig,
    """
    # get the hemi networks_names
    hemi_networks = list(data_FC_parcel.columns.values)

    # get the unique hemi networks_names
    hemi_network_unique = np.unique(data_FC_parcel.columns.values)

    if corr_value == 'z':
        # replace diagonal with zero values
        np.fill_diagonal(data_FC_parcel.values, 0)

        # transform z to r
        np_FC_parcel = np.tanh(data_FC_parcel.values)

        # todo change it
        np.fill_diagonal(np_FC_parcel,1)

    else:
        np_FC_parcel = data_FC_parcel.values

    #
    data_unique = np.unique(np_FC_parcel)
    vmax = np.max(data_unique)
    vmin = np.min(data_unique)

    # 1 is the diagonal
    if vmax == 1:
        vmax = data_unique[-2]
    # 0 is also the diagonal
    if vmin == 0:
        vmin = data_unique[1]

    plotting.plot_mod_heatmap(data = np_FC_parcel, communities = hemi_networks, figsize=(14, 12), xlabels=hemi_network_unique, ylabels=hemi_network_unique, vmin=vmin, vmax=vmax, cmap= cmap, mask_diagonal=False)
    plt.savefig(file_fig)
    plt.close()

    # add title of the figure
    img = Image.open(file_fig)
    image_editable = ImageDraw.Draw(img)
    image_editable.text((title_position, 100), title, (0, 0, 0), font=font)
    img.save(file_fig, "PNG")

def plot_FC_matrix_network(data_FC_parcel_mean_T_mean, title, fig_networks):
    """
    :param data_FC_parcel_mean_T_mean: is the 34 * 34 FC networks_names or 17 * 17
    :param fig_networks: the fig file_hctsa
    :return: None
    """
    df = data_FC_parcel_mean_T_mean
    networks_all = list(df.columns.values)

    f = plt.figure(figsize=(20, 20))
    plt.matshow(df.values, cmap=cmap, fignum=f.number)
    plt.xticks(np.arange(0,len(networks_all)), networks_all, fontsize=14, rotation=90)
    plt.yticks(np.arange(0,len(networks_all)), networks_all, fontsize=14)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.title(title, fontsize=20)
    plt.savefig(fig_networks)
    plt.close()

def load_matrix_1d_file(file_400_1d,network_num = 34,parcel_name=False, sort=True):
    """
       :param file_400_1d: is the 400 * 1 value - z values; can be ptseries.nii, xlsx, array
                                 the order is network 1, 2, 3,,,400
       :param network_num: can be 34, LH 17 and RH 17; 17: Bilateral
       :param sort: whether sort the column name and index name; true, sort; false no sort
       :return: 400 * 1, the order is LH vis 1,2,3 .. DMN 1, 2, 3; RH Vis 1, 2, 3 .. DMN 1, 2, 3
       """

    if type(file_400_1d) == np.ndarray:
        # in this case, the file_stat_parcel_400 is an array, the shape is 400 * 400
        data_400_1d = pd.DataFrame(data=file_400_1d)

    elif type(file_400_1d) == pd.core.frame.DataFrame:
        data_400_1d = file_400_1d

    elif file_400_1d.endswith('.xlsx'):
        data_400_1d = pd.read_excel(file_400_1d, header=None)

    elif file_400_1d.endswith('.ptseries.nii'):
        # load the file, get the data and convert it to df
        data_400_1d = pd.DataFrame(data=nib.load(file_400_1d).get_fdata()[0])

    # load the data_atlas
    data_atlas = load_atlas(file_atlas)

    if parcel_name  and network_num == 34:
        col_add = col_hemi_network_parcel
    elif parcel_name  and network_num == 17:
        col_add = col_network_parcel
    elif  not parcel_name and network_num == 34:
        col_add = col_hemi_network
    elif  not parcel_name and network_num == 17:
        col_add = col_network

    column_names = list(data_atlas[col_add].values)

    # add one column
    data_400_1d[col_hemi_network] = column_names

    # set it as index
    data_400_1d.set_index(col_hemi_network, inplace=True)

    # sort it by column names and index
    if sort:
        data_400_1d.sort_index(inplace=True)
        data_400_1d.sort_index(axis=1, inplace=True)

    return data_400_1d





#%% test them

#
# # plot it
#
# file_parcel_400 = file_FC_parcel_400_1
#
# # load the atlas
# data_atlas = load_atlas(file_atlas)
#
# # load the FC matrix
# data_FC_parcel_for_34 = load_matrix_file(file_parcel_400, network_num=34)
#
# # plot the whole matrix
# file_fig_whole = '/home/xiuyi/Data/Task_Gradient/Nifti/fMRIPrep_cifti_nifti_xcp_abcd/smooth_6_filter_min_0.01_0.08_36P_cifti/xcp_abcd_FC_gradient_timescale_group/FC/test.png'
# plot_matrix_whole(data_parcel = data_FC_parcel_for_34, file_fig =file_fig_whole, title='task-FeatureMatching_FC', title_position=350, corr_value ='z', )
#
# # convert 417 to 34
# data_FC_parcel_34 =  convert_FC_417_networks(data_FC_parcel_for_34, corr_value ='z')
#
# # plot it
# fig_FC_networks_34 = '/home/xiuyi/Data/Task_Gradient/Nifti/fMRIPrep_cifti_nifti_xcp_abcd/smooth_6_filter_min_0.01_0.08_36P_cifti/xcp_abcd_FC_gradient_timescale_group/FC/ses-01_task-FeatureMatching_demean_parcel_merge_FC_z_34.png'
#
# title_network_34 = 'feature matching FC networks_names 34'
#
# plot_FC_matrix_network(data_FC_parcel_34,title = title_network_34, fig_networks = fig_FC_networks_34)
#
#
# data_FC_parcel_for_17 = load_matrix_file(file_parcel_400, network_num=17)
#
# # convert 417 to 34
# data_FC_parcel_34 =  convert_FC_417_networks(data_FC_parcel_for_17, corr_value ='z')
#
# # plot it
# fig_FC_networks_17 = '/home/xiuyi/Data/Task_Gradient/Nifti/fMRIPrep_cifti_nifti_xcp_abcd/smooth_6_filter_min_0.01_0.08_36P_cifti/xcp_abcd_FC_gradient_timescale_group/FC/ses-01_task-FeatureMatching_demean_parcel_merge_FC_z_17.png'
#
# title_network_17 = 'feature matching FC networks_names 17'
#
# plot_FC_matrix_network(data_FC_parcel_mean_T_mean = data_FC_parcel_34,title = title_network_17,fig_networks = fig_FC_networks_17)
