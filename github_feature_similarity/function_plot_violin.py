# plot the violin figure

from matplotlib import pyplot
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import itertools
from scipy import stats
import seaborn as sns
from function_convert_matrix_417_to_17 import convert_mat_417_N_networks, load_atlas
import numpy as np
import pandas as pd
from constants import cols_short, shortlabels

font_file = '/home/xiuyi/Data/Atlas/Playfair_Display/arial/arial.ttf'
file_atlas = '/home/xiuyi/Data/Atlas/Parcellation_Kong_2022/Schaefer_417.xlsx'
data_atlas = load_atlas(file_atlas)
cols_short=['Vis-A','Vis-B','Vis-C','Aud','SM-A','SM-B','VAN-A', 'VAN-B', 'DAN-A','DAN-B','Cont-A','Cont-B','Cont-C','Lang','DMN-A','DMN-B','DMN-C']

def prepare_data_for_plot_violin(data, isacending=True, value_name='value'):
    """
    prepare a df for plotting the data as a violin
    :param data: shape: N (subjects) * 400 (parcels)
    :return: a df and a order of the networks
    """
    # convert N * 400 to N * 17
    data_for_violin = np.zeros((data.shape[0], 17))

    for i in range(data.shape[0]):
        data_network_each = convert_mat_417_N_networks(data[i,:],network_num = 17, data_atlas=data_atlas)

        data_for_violin[i,:] = data_network_each.values.ravel()

    df_data_for_violin = pd.DataFrame(data = data_for_violin, columns = cols_short)

    df_data_for_violin['participant'] = np.arange(data.shape[0])

    # convert it from wide to long
    df_data_for_violin_long = pd.melt(df_data_for_violin, id_vars='participant', value_vars=cols_short, var_name='network', value_name=value_name)

    # Using pandas methods and slicing to determine the order of networks by increasing mean
    order_mean = list(df_data_for_violin_long.groupby(by=["network"])[value_name].mean().sort_values(ascending=isacending).index.values)

    return  df_data_for_violin_long, order_mean

def prepare_data_for_plot_violin_v2(data,  isacending=False, value_name='value'):
    """
    prepare a df for plotting the data as a violin, and list the each datapoint
    :param data: shape: 1 * 400 (parcels)
    :return: a df and a order of the networks, 
    """

    column_names = list(data_atlas['network'].values)
    colums_names_short = [shortlabels[x] for x in column_names]

    if data.ndim > 1: data=data.ravel()
 
    df_long = pd.DataFrame({'value':data})
    df_long['network'] = colums_names_short
    df_long.set_index('network', inplace=False)

    order_mean = list(df_long.groupby(by=["network"])[value_name].mean().sort_values(ascending=isacending).index.values)

    return  df_long, order_mean

def ttest_violin(df, order, pairlist, x_variable, y_variable, groupnum=0, adjacent=True, method= 't-test_paired'):
    
    # do t-test for only adjacent pair under order, or all unique comparison between group
    if adjacent:
        order_filter = [net for net in order if net in pairlist]
        stats_pairs = [(order_filter[k-1], order_filter[k]) for k in range(1,len(order_filter))]
    else:
        stats_pairs = itertools.combinations(pairlist, 2)

    N1=list(); N2=list(); t_value=list(); p_value=list()
    for (x1,x2) in stats_pairs:
        y1, y2 = df.loc[df[x_variable]==x1, y_variable], df.loc[df[x_variable]==x2, y_variable]
        if method == 't-test_paired':
            t, p = stats.ttest_rel(y1, y2)
        elif method == 't-test_ind':
            t, p = stats.ttest_ind(y1, y2)
        else:
            raise ValueError('method only accept t-test_paired or t-test_ind')

        N1.append(x1); N2.append(x2); t_value.append(t); p_value.append(p)
    
    df_stats = pd.DataFrame({"N1":N1, "N2":N2, "t_value":t_value, "p_value":p_value, "group":groupnum})

    return df_stats

def star_sig(p):
    # change p value to star string
    if   p<.001:
        return '***'
    elif p< .01:
        return '**'
    elif p< .05:
        return '*'
    else:
        return 'ns'

def plot_violin_v3(df,x_variable,y_variable,fig_size, order, figfile,
                    title=None, x_label='', y_label='', fontsize_title=10,fontsize_label=10, palette=None,
                    y_lim=None, y_ticks=None):

    """
    this is for plotting the confusion matrix, only including the top 7 networks
    :param df: is the dataframe for plot: columns:x_variable,y_variable
    :param x_variable: the xlabel
    :param y_variable: y values
    :param fig_size:
    :param y_label:
    :param order: the order of x_variable
    :param palette:
    :param title:
    :param figfile:
    :return:
    """
    fontprob_label=fm.FontProperties(fname=font_file, size=fontsize_label)

    a4_dims = fig_size
    fig, ax = plt.subplots(figsize=a4_dims)

    plt.rcParams.update({'font.family':'Arial', 'font.size': fontsize_label})
    sns.violinplot(ax = ax,x=x_variable, y=y_variable, data=df, order=order, linewidth=0.8,palette=palette)
    sns.stripplot(ax = ax, x=x_variable, y=y_variable, data=df, order=order, palette=palette, jitter=True, dodge=True, size=3, linewidth=0.5)
    
    ax.set_xticklabels(order,fontproperties=fontprob_label,rotation=90)
    plt.setp(ax.get_yticklabels(), fontproperties=fontprob_label)
    #ax.tick_params(axis='y', labelsize=fontsize_label)
    
    if title is not None:
        ax.set_title(title,fontsize=fontsize_title,fontweight='bold', y=1)
    
    
    ax.set_xlabel(x_label, fontproperties=fontprob_label)
    ax.set_ylabel(y_label, fontproperties=fontprob_label, x=-0.2)
    
    
    if y_lim is not None:
        ax.set_ylim(y_lim)
    if y_ticks is not None:
        ax.set_yticks(y_ticks)
   

    # Only show ticks on the left and bottom spines
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plt.locator_params(axis='y', nbins=5) 

    plt.subplots_adjust(left=0.12, right=0.99, bottom=0.2, top=0.92, wspace=0.0, hspace=0)
    
    #save
    plt.savefig(figfile,dpi=600)
    figfile2=figfile[:-4]+'.pdf'
    plt.savefig(figfile2,format="pdf")
    plt.close()

def plot_violin_v4(ax, df,x_variable,y_variable,  fig_size, order, figfile,
                    title=None, x_label='', y_label='', fontsize_title=10,fontsize_label=10, palette=None,
                    y_lim=None, y_ticks=None,y_ticklabels=None):

    """
    this is for plotting the confusion matrix, compared to v4, this added the statistical significance given pairlist and p-values.
    :param df: is the dataframe for plot: columns:x_variable,y_variable
    :param x_variable: the xlabel
    :param y_variable: y values
    :param fig_size:
    :param y_label:
    :param order: the order of x_variable
    :param palette:
    :param title:
    :param figfile:
    :return:
    """
    fontprob_label=fm.FontProperties(fname=font_file, size=fontsize_label)

    # plot 
    sns.violinplot(ax = ax,x=x_variable, y=y_variable, data=df, order=order, linewidth=0.8,palette=palette, cut=2)
    sns.stripplot(ax = ax, x=x_variable, y=y_variable, data=df, order=order, palette=palette, jitter=True, dodge=True, size=4, edgecolor='white', linewidth=0.5)
    
    #labeling
    ax.set_xticklabels(order,fontproperties=fontprob_label,rotation=90)
    plt.setp(ax.get_yticklabels(), fontproperties=fontprob_label)

    #labeling
    ax.set_xlabel(x_label, fontproperties=fontprob_label)
    ax.set_ylabel(y_label, fontproperties=fontprob_label, x=-0.2)

    if title is not None:
        ax.set_title(title,fontsize=fontsize_title,fontweight='bold', y=1)
    if y_lim is not None:
        ax.set_ylim(y_lim)
    if y_ticks is not None:
        ax.set_yticks(y_ticks)
    if y_ticklabels is not None:
        ax.set_yticklabels(y_ticklabels)
   

    # Only show ticks on the left and bottom spines
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    #plt.locator_params(axis='y', nbins=5) 


    return ax, fontprob_label


def plot_violin_v5(figfile, ax, df,x_variable,y_variable, df_stats, p_variable,  order,
                    title=None, x_label='', y_label='', fontsize_title=10,fontsize_label=10, palette=None,
                    y_lim=None, y_ticks=None,y_ticklabels=None):

    """
    this is for plotting the confusion matrix, only including the top N networks
    :param df: is the dataframe for plot: columns:x_variable,y_variable
    :param x_variable: the xlabel
    :param y_variable: y values
    :param fig_size:
    :param y_label:
    :param order: the order of x_variable
    :param palette:
    :param title:
    :param figfile:
    :return:
    """
    fontprob_label=fm.FontProperties(fname=font_file, size=fontsize_label)
    # fontprob_label_s=fm.FontProperties(fname=font_file, size=fontsize_label-1)
    fontprob_label_s=fm.FontProperties(fname=font_file, size=fontsize_label)
    # plot
    sns.violinplot(ax = ax,x=x_variable, y=y_variable, data=df, order=order, linewidth=0.8,palette=palette, cut=2)
    sns.stripplot(ax = ax, x=x_variable, y=y_variable, data=df, order=order, palette=palette, jitter=True, dodge=True, size=4, edgecolor='k', linewidth=0.5)
    

    #labeling
    ax.set_xticklabels(order,fontproperties=fontprob_label,rotation=90)
    plt.setp(ax.get_yticklabels(), fontproperties=fontprob_label)

    #labeling
    ax.set_xlabel(x_label, fontproperties=fontprob_label)
    ax.set_ylabel(y_label, fontproperties=fontprob_label, x=-0.2)

    if title is not None:
        ax.set_title(title,fontsize=fontsize_title, y=1.05)
    if y_lim is not None:
        ax.set_ylim(y_lim)
    if y_ticks is not None:
        ax.set_yticks(y_ticks)
    if y_ticklabels is not None:
        ax.set_yticklabels(y_ticklabels)
    
    y_min, y_max=ax.get_ylim()
    
    # add the star statistical annotation
    line_width=1
    y, h, col = y_max, (y_max-y_min)*0.02 , 'k'
    for index, row in df_stats.iterrows():
        x1, x2 = order.index(row['N1'])+0.05, order.index(row['N2'])-0.05  # x coordinates of two networks
        sig_str = star_sig(row[p_variable])
        
        ax.plot([x1, x1, x2, x2], [y-h, y, y, y-h], lw=line_width, c=col)
        if sig_str == 'ns':
            # baihan ori value
            # ax.text((x1+x2)*.5, y-h*0.4, sig_str, ha='center', va='bottom', color=col, fontproperties=fontprob_label_s)
            ax.text((x1+x2)*.5, y-h*0.01, sig_str, ha='center', va='bottom', color=col, fontproperties=fontprob_label_s)
        else:
            # baihan ori value
            # ax.text((x1+x2)*.5, y-h*1.5, sig_str, ha='center', va='bottom', color=col, fontproperties=fontprob_label)
            ax.text((x1+x2)*.5, y-h*0.1, sig_str, ha='center', va='bottom', color=col, fontproperties=fontprob_label)

    # Only show ticks on the left and bottom spines
    # ax.spines.right.set_visible(False)
    # ax.spines.top.set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    #plt.locator_params(axis='y', nbins=5)

    plt.savefig(figfile, dpi=300)
    plt.close()
        
    return ax, fontprob_label

