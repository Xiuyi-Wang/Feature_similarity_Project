import numpy as np
from scipy.stats import pearsonr
from scipy.stats import spearmanr

def make_matrix_symmetric(data_mat):
    """
    make the matrix symmetric, copy the upper triangle to lower trianlge
    :param data_mat:
    :return:
    """
    data_mat_sym = np.maximum(data_mat, data_mat.transpose())
    return data_mat_sym


def calculate_correlation_symmetric_matrix(mat_1 ,mat_2 ,perm_num ,method='pearsonr'):
    """
    calculate the correlation of two symmetric matrix
    :param mat_1: matrix_1
    :param mat_2: matrix_2
    :param perm_num: permutation_num
    :param method: pearsonr, or spearmanr
    :return: r_real, p_real, p_perm
    """
    # get the index of the low triangle of the matrix
    il = np.tril_indices(mat_1.shape[0], -1)

    # only choose the values of the low triangle
    mat_1_tril = mat_1[il]
    mat_2_tril = mat_2[il]

    # calculate correlation
    r_real, p_real = pearsonr(mat_1_tril ,mat_2_tril)

    print (r_real, p_real)

    # run the permutation
    r_perm_all = np.zeros((perm_num))
    for i in range(perm_num):
        np.random.shuffle(mat_1_tril)
        r_perm, p_perm =  pearsonr(mat_1_tril ,mat_2_tril)
        r_perm_all[i] =   r_perm

    np.sort(r_perm_all)

    # find the corresponding p
    if r_real > 0:
        count = np.count_nonzero(r_perm_all > r_real)
    elif r_real < 0:
        count = np.count_nonzero(r_perm_all < r_real)

    p_perm_all = count /perm_num

    print (r_real ,p_real ,p_perm_all, count, perm_num)

    return r_real ,p_real ,p_perm_all, count, perm_num

def calculate_correlation_symmetric_matrix_v2(mat_1 ,mat_2,method):
    """
    calculate the correlation of two symmetric matrix
    :param mat_1: matrix_1
    :param mat_2: matrix_2
    :param perm_num: permutation_num
    :param method: pearsonr, or spearmanr
    :return: r_real, p_real, p_perm
    """
    # get the index of the low triangle of the matrix
    il = np.tril_indices(mat_1.shape[0], -1)

    # only choose the values of the low triangle
    mat_1_tril = mat_1[il]
    mat_2_tril = mat_2[il]

    # calculate correlation
    if method == 'pearson':
        r_real, p_real = pearsonr(mat_1_tril ,mat_2_tril)
    elif method == 'spearman':
        r_real, p_real = spearmanr(mat_1_tril, mat_2_tril)

    return r_real, p_real, mat_1_tril.shape[0]