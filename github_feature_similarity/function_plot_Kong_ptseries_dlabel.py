# plot the Kong map

# the raw input is 400 values

# the output is a figure


import numpy as np
import os
from os import system
from os.path import join, exists
from function_edit_scene_figures import add_title, add_color_bar
from wbplot_images import write_parcellated_image
import nibabel as nib
from zipfile import ZipFile
import matplotlib.font_manager as fm
from PIL import Image, ImageFont, ImageDraw
from function_edit_scene_figures_baihan import add_colorbar2

num_decimal = 2
path_wb = '/home/xiuyi/Software/workbench/workbench/bin_linux64/wb_command'
font_file = '/home/xiuyi/Data/Atlas/Playfair_Display/arial/arial.ttf'
titlefont = ImageFont.truetype(font=font_file,size=100)
fontprop = fm.FontProperties(fname=font_file,size=36)


def save_ptseries_file(array, file_map):
    """
    generate pconn file_hctsa
    :param array: 400, shape (1,400)
    :param file_map: ptseries.nii file_hctsa
    :return:
    """
    import nibabel as nib
    import os

    # step 2: read the excel file_hctsa and then save the data
    cifti_template_file = '/home/xiuyi/Data/Atlas/Parcellation_Kong_2022/template.ptseries.nii'
    cifti_template = nib.load(cifti_template_file)

    new_img = nib.Cifti2Image(array, header=cifti_template.header, nifti_header=cifti_template.nifti_header)
    new_img.to_filename(file_map)

def save_dlabel_Kong (array,file_output):
    """
    :param array: is a list or array that includes that parcel IDs that to be plotted
    :param file_output: is the dlabel.nii file_hctsa, networks that need to be plotted are the network ID, others are np.nan
    :return:
    """
    import nibabel as nib
    from os.path import  join
    import numpy as np

    path_template = '/home/xiuyi/Data/Atlas/Parcellation_Kong_2022'
    filename = 'Schaefer2018_400Parcels_Kong2022_17Networks_order.dlabel.nii'

    file = join(path_template,filename)

    data = nib.load(file).get_fdata()

    # replace the value, only keep the values in the array
    data_new = np.where(np.in1d(data,array), data, np.nan)

    cifti_template = nib.load(file)

    new_img = nib.Cifti2Image(data_new, header=cifti_template.header, nifti_header=cifti_template.nifti_header)

    new_img.to_filename(file_output)

def plot_Kong_parcellation(array, fig, title, title_position=180):
    """
    :param array: is 400 values of the network, shape is (1,400)
    :param file_output: the name of png
    :return:
    """

    path_output = '/home/xiuyi/anaconda3/lib/python3.8/site-packages/wbplot/data/kong_ptseries'
    os.makedirs(path_output, exist_ok = True)

    # The original scene file you created
    path_base = '/home/xiuyi/Data/Atlas/scene_Kong'
    filename_scene = 'Schaefer_417.scene'
    folder_scene = 'Schaefer_417_inflated'

    # the original ptseries file you created
    filename = 'ImageParcellated.ptseries.nii'
    file_map = os.path.join(path_output, filename)

    file_scene_1 = os.path.join(path_base, filename_scene)
    file_scene_folder = os.path.join(path_base, folder_scene)


    # check whether scene file exists, if not, copy them
    if not os.path.exists(os.path.join(path_output,folder_scene )):
        cmd_1 = "cp -r {} {}".format(file_scene_folder, path_output)
        system(cmd_1)

    if not os.path.exists(os.path.join(path_output,filename_scene)):
        cmd_2 = "cp  {} {}".format(file_scene_1, path_output)
        system(cmd_2)

    # save the ptseries file
    save_ptseries_file(array,file_map)

    scene_file = os.path.join(path_output,filename_scene)
    scene=1
    width = 1263
    # height = 835
    height = 1200

    # copy scene file
    cmd = path_wb + ' -show-scene "{}" {} "{}" {} {}'.format(scene_file, scene,  fig, width, height)

    system(cmd)

    # add title
    add_title(fig, title, title_position)

    # add color bar
    data = array

    # divide it to positive and negative values
    data_neg = data[data < 0]
    data_pos = data[data > 0]

    # find the positive values and negative values (extreme)
    # these values would be used when adding color bar
    if data_pos.shape[0] != 0:
        vmin_pos_max = np.round(data_pos.max(), num_decimal)
        vmin_pos_min = np.round(data_pos.min(), num_decimal)
    else:
        vmin_pos_max = 0
        vmin_pos_min = 0
    if data_neg.shape[0] != 0:
        vmin_neg_max = np.round(data_neg.max(), num_decimal)
        vmin_neg_min = np.round(data_neg.min(), num_decimal)
    else:
        vmin_neg_max = 0
        vmin_neg_min = 0

    add_color_bar(fig, vmin_neg_min, vmin_neg_max, vmin_pos_min, vmin_pos_max, color='psych')


def plot_Kong_parcellation_2(data, fig, cmap= 'nipy_spectral', title=None, title_position=180):
    """
    :param array: is 400 values of the network, shape is (1,400)
    :param file_output: the name of png
    :return:
    """

    if type(data) is np.ndarray:
        pscalars = data
    elif isinstance(data, str):
        d = nib.load(data).get_fdata()
        pscalars = d[0]
    else:
        raise ValueError('data must be a data array or a cifti file ')
    
    
    # [CHANNGE]
    # path_base = '/home/han/FC_400parcel_distance/scene/scene_Kong_dlabel'
    path_base = '/home/xiuyi/Data/Atlas/scene_Kong_dlabel'
     # The original scene file you created
    scene_zip_file=join(path_base,'Schaefer_417_dlabel.zip')
    filename_scene = 'Schaefer_417_dlabel.scene'
    temp_name = 'ImageParcellated.dlabel.nii'

    temp_dir = os.path.dirname(fig)
    
    # copy the scene file & SchaeferParcellations directory to the
    # temp directory as well
    with ZipFile(scene_zip_file, "r") as z:  # unzip to temp dir
        z.extractall(temp_dir)
    scene_file = join(temp_dir, 'scene_Kong_dlabel', filename_scene)
    if not exists(scene_file):
        raise RuntimeError(
            "scene file was not successfully copied to {}".format(scene_file))

    # Write `pscalars` to the neuroimaging file which is pre-loaded into the
    # scene file, and update the colors for each parcel using the file metadata
    temp_cifti = join(temp_dir, temp_name)
    temp_cifti2 = join(temp_dir,'scene_Kong_dlabel',temp_name)
    write_parcellated_image(data=pscalars, fout=temp_cifti,  cmap=cmap)

    # overwrite the original files
    cmd='cp -vf %s %s'%(temp_cifti,temp_cifti2)
    system(cmd)
    
    # copy scene file
    scene=1
    width = 1263
    height = 1200   # height = 835
    cmd = path_wb + ' -show-scene "{}" {} "{}" {} {}'.format(scene_file, scene, fig, width, height)
    system(cmd)
    
    # remove the copied scene folder
    cmd = 'rm -r %s'%(join(temp_dir,'scene_Kong_dlabel'))
    system(cmd)

    # add title
    add_title(fig, title, title_position)

    # divide it to positive and negative values
    data_neg = data[data < 0]
    data_pos = data[data > 0]

    # find the positive values and negative values (extreme)
    # these values would be used when adding color bar
    
    if data_pos.shape[0] != 0:
        vmin_pos_max = np.round(data_pos.max(), num_decimal)
        vmin_pos_min = np.round(data_pos.min(), num_decimal)
    else:
        vmin_pos_max = 0
        vmin_pos_min = 0
    if data_neg.shape[0] != 0:
        vmin_neg_max = np.round(data_neg.max(), num_decimal)
        vmin_neg_min = np.round(data_neg.min(), num_decimal)
    else:
        vmin_neg_max = 0
        vmin_neg_min = 0
    add_colorbar2(pscalars, fig, cmap, orientation='horizontal')

    # add_color_bar(fig, vmin_neg_min, vmin_neg_max, vmin_pos_min, vmin_pos_max, color='nipy_spectral')
    


def plot_Kong_dlabel(array, fig, title, title_position,title_position_vertical):
    """
    :param array: is the parcel id need to be plotted [1,3,5]
    :param file_output: the name of png
    :return:
    """

    path_output = '/home/xiuyi/anaconda3/lib/python3.8/site-packages/wbplot/data/kong_dlabel'
    # path_output = '/home/xiuyi/Data/Task_Gradient/Nifti/fMRIPrep_cifti_nifti_xcp_abcd/smooth_6_filter_min_0.01_0.08_36P_cifti/ptseries_individual_parcellation_32k_task_regression_residual_FC_temporal_ac_xDF_indi_parcel_based_FC_stats/Association/dlabel'
    os.makedirs(path_output, exist_ok = True)

    # The original scene file you created
    path_base = '/home/xiuyi/Data/Atlas/scene_Kong_dlabel'
    filename_scene = 'Schaefer_417_dlabel.scene'
    folder_scene = 'Schaefer_417_inflated'

    # the original ptseries file you created
    filename = 'ImageParcellated.dlabel.nii'
    file_map = os.path.join(path_output, filename)

    file_scene_1 = os.path.join(path_base, filename_scene)
    file_scene_folder = os.path.join(path_base, folder_scene)


    # check whether scene file exists, if not, copy them
    if not os.path.exists(os.path.join(path_output,folder_scene )):
        cmd_1 = "cp -r {} {}".format(file_scene_folder, path_output)
        system(cmd_1)

    if not os.path.exists(os.path.join(path_output,filename_scene)):
        cmd_2 = "cp  {} {}".format(file_scene_1, path_output)
        system(cmd_2)

    # save the ptseries file
    save_dlabel_Kong(array, file_map)

    scene_file = os.path.join(path_output,filename_scene)
    scene=1
    width = 1263
    # height = 835
    height = 1100

    # copy scene file
    cmd = path_wb + ' -show-scene "{}" {} "{}" {} {}'.format(scene_file, scene,  fig, width, height)

    system(cmd)

    # add title
    add_title(fig, title,title_position,title_position_vertical)

# title = 'test'
# array = np.array(np.arange(1,401)).reshape(1, 400)
# fig = '/home/xiuyi/anaconda3/lib/python3.8/site-packages/wbplot/data/kong_ptseries/test_2.png'
# plot_Kong_parcellation(array, fig,title)
#
# array = [1,3,5]
# fig = '/home/xiuyi/anaconda3/lib/python3.8/site-packages/wbplot/data/kong_dlabel/test_2.png'
#
# plot_Kong_dlabel(array, fig,title)
