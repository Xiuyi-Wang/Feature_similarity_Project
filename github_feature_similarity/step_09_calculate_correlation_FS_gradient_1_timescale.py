# calculate the spatial correlation of HCP rest, FC and FS gradients and
# after spin permutation
import nibabel as nib
from scipy.stats import pearsonr
import os
import pandas as pd
import numpy as np
from brainsmash.mapgen.base import Base
from brainsmash.mapgen.stats import pearsonr, pairwise_r
import matplotlib.pyplot as plt
from scipy import stats
from brainsmash.mapgen.stats import nonparp
import matplotlib.pyplot as plt
from scipy import stats
import mat73
from scipy.io import loadmat
from function_plot_Kong_ptseries_dlabel import plot_Kong_parcellation_2


# calculate correlation between timescale and FS gradient 2
file_AC = '/home/xiuyi/Data/HCP/99_Results/timescale/HCP_rest_AC_baihan/group_task-rest_demean_parcel_ac_timescale.mat'
path_FS = '/home/xiuyi/Data/HCP/99_Results/hctsa_timeseries_fs/gradient_HCP_rest_FS/kernel_None_embedding_dm'

fig_AC = '/home/xiuyi/Data/HCP/99_Results/timescale/HCP_rest_AC_baihan/group_task-rest_demean_parcel_ac_timescale.png'

# run spin permutation
sac = '#377eb8'  # autocorr-preserving
rc = '#e41a1c'  # randomly shuffled
permutation_num = 5000
path_output = '/home/xiuyi/Data/HCP/99_Results/HCP_rest_FC_FS_gradient_correlation'

os.makedirs(path_output,exist_ok=True)

# distance file and data
file_dist_L  = '/home/xiuyi/Data/HCP/03_pconn_data_group/HCPA_distance_parcel_based_L.mat'
file_dist_R  = '/home/xiuyi/Data/HCP/03_pconn_data_group/HCPA_distance_parcel_based_R.mat'

key_dist = 'geodesic_distance_parcel'

df_pconn_L = loadmat(file_dist_L)[key_dist]
df_pconn_R = loadmat(file_dist_R)[key_dist]


# split the data into left and right and then sort it by parcel id

data_AC_all = mat73.loadmat(file_AC)['time_scale_all']

data_AC = np.mean(data_AC_all,axis=0)

title = 'HCPA rest time scale'
# plot_Kong_parcellation_2(data_AC, fig_AC, 'nipy_spectral', title, title_position=100)
plot_Kong_parcellation_2(data_AC, fig_AC, 'jet', title, title_position=100)

for j in range(1,4):
    file_FS_gradient = os.path.join(path_FS, 'HCP_rest_FS_gradient_%s_kernel-None_embedding-dm_flip.ptseries.nii' % (j))

    data_FS_grad = nib.load(file_FS_gradient).get_fdata()[0]

    r, p = stats.pearsonr(data_AC,data_FS_grad)
    print ( j ,r, p)

    for hemi in ['L','R']:

        fig_file = os.path.join(path_output,'Correlation_time_scale_FS_grad_%s_%sH.png'%(j,hemi))
        if hemi == 'L':
            # load parcellated structural neuroimaging maps
            data_FC_grad_half = data_AC[0:200]
            data_FS_grad_half = data_FS_grad[0:200]
            df_pconn = df_pconn_L
        elif hemi =='R':
            # load parcellated structural neuroimaging maps
            data_FC_grad_half = data_AC[200::]
            data_FS_grad_half = data_FS_grad[200::]
            df_pconn = df_pconn_R

        # instantiate class and generate 1000 surrogates
        generator = Base(x=data_FC_grad_half, D=df_pconn, resample=True)
        surrogate_maps = generator(n=permutation_num)

        # compute the Pearson correlation between each surrogate FC_gradient_1 map
        # and the empirical cortical FS_gradient_2 map
        surrogate_brainmap_corrs = pearsonr(data_FS_grad_half, surrogate_maps).flatten()

        # print (surrogate_brainmap_corrs)
        surrogate_pairwise_corrs = pairwise_r(surrogate_maps, flatten=True)

        # Repeat using randomly shuffled surrogate FC_gradient_1 maps:

        bins = np.linspace(-1, 1, 51)  # correlation b

        # this is the empirical statistic we're creating a null distribution for
        test_stat = stats.pearsonr(data_FC_grad_half, data_FS_grad_half)[0]

        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_axes([0.2, 0.25, 0.6, 0.6])  # autocorr preserving
        ax2 = ax.twinx()  # randomly shuffled

        # plot the data
        ax.axvline(test_stat, 0, 0.8, color='k', linestyle='dashed', lw=1)
        ax.hist(surrogate_brainmap_corrs, bins=bins, color=sac, alpha=1,
            density=True, clip_on=False, zorder=1)


        # make the plot nice...
        ax.set_xticks(np.arange(-1, 1.1, 0.5))
        ax.spines['left'].set_color(sac)
        ax.tick_params(axis='y', colors=sac)
        ax2.spines['right'].set_color(rc)
        ax2.tick_params(axis='y', colors=rc)
        ax.set_ylim(0, 2)
        ax2.set_ylim(0, 6)
        ax.set_xlim(-1, 1)
        [s.set_visible(False) for s in [
            ax.spines['top'], ax.spines['right'], ax2.spines['top'], ax2.spines['left']]]

        ax.text(0.97, 1.03, 'SA-preserving', ha='right', va='bottom',
            color=sac, transform=ax.transAxes)
        ax.text(test_stat, 1.65, "real r %s"%(np.round(test_stat,3)), ha='center', va='bottom')
        ax.text(0.5, -0.2, "Pearson correlation %sH"%(hemi),
            ha='center', va='top', transform=ax.transAxes)
        ax.text(-0.3, 0.5, "Density", rotation=90, ha='left', va='center', transform=ax.transAxes)
        plt.savefig(fig_file, dpi=300)

        plt.close()
