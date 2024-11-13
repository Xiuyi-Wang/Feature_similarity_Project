import os
import pandas as pd
import numpy as np
import mat73
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import joblib
import sys
sys.modules['sklearn.externals.joblib'] = joblib
from scipy.stats import pearsonr
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler

def load_normalized_feature_vectors(file_features_N,file_label,file_operations):
    """
    load the normalized feature vectors and labels for classification
    file_operations: is the operations of the file_features_N
    :param file_features_N:
    :param file_label:
    :param file_operations: is the operations of the file_features_N
    :return: data: 1600 (parcels) * ~7000 (features), labels (1600 * 1, network of each parcel), operations: N (feature number) *4
    """
    # read the label file and get the label of each sample
    data_label = pd.read_excel(file_label)
    labels_each = data_label['network_id']
    labels = np.array(list(labels_each) * 4)

    # load data of feature vectors
    data_all = mat73.loadmat(file_features_N)
    data = data_all['TS_DataMat']

    # load the data of operations
    data_operations = pd.read_excel(file_operations)

    if data.shape[1] != data_operations.shape[0]:
        print ('error: features and operations  do not match')

    return data, labels, data_operations


def choose_part_features(data, feature_part_num):
    """
    choose the first N features that have many distinctive values
    :param data: 1600 * (~7000) features
    :param feature_part_num: the N features you would like to select
    :return: part of the data
    """
    df = pd.DataFrame(data, columns=list(np.arange(1, data.shape[1] + 1)))
    distinct_counter = df.apply(lambda x: len(x.unique()))

    # sort the distinct_counter based on the unique number
    distinct_counter.sort_values(ascending=False, inplace=True)

    cols_index = list(distinct_counter.index)

    # choose the first * features for classification
    data_part = df[cols_index[0:feature_part_num]].values

    return data_part

def svm_classify_kfold(X, y,kernel,scaler_stats =False):
    """
    define a function to run multiclass classification
    :param X: features set: 1600 * 7000; datatype np
    :param y: class label: 1600 * 1; datatype np
    :param kernel: the kernel of svm
    :return: classification accuracy, f1, confusion matrix cm
    """
    # create a svm with Polynomial kernel
    kfold_num = 5
    kf = KFold(n_splits=kfold_num, shuffle=True)

    # create a np to store the classification accuracy/confusion matrix of each fold
    stat_all = np.zeros((kfold_num, 2))
    np_confusion_all = np.zeros((kfold_num,np.unique(y).shape[0],np.unique(y).shape[0]))
    i = 0

    for train_index, test_index in kf.split(X):

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        if scaler_stats:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        if kernel == 'linear':

            # svm_classifier = svm.SVC(kernel='linear', degree=3, C=5, class_weight='balanced').fit(X_train, y_train)
            svm_classifier = svm.SVC(kernel='linear', degree=3, C=5).fit(X_train, y_train)

        elif kernel == 'poly':
            # svm_classifier = svm.SVC(kernel='poly', degree=3, C=5,class_weight='balanced').fit(X_train, y_train)
            svm_classifier = svm.SVC(kernel='poly', degree=3, C=5).fit(X_train, y_train)

        elif kernel == 'rbf':
            svm_classifier = svm.SVC(kernel='rbf', gamma=5, C=0.001).fit(X_train, y_train)

        # test the classifier using the test data set
        y_prediction = svm_classifier.predict(X_test)

        # calculate the accuracy and f1 scores
        accuracy = accuracy_score(y_test, y_prediction)
        f1 = f1_score(y_test, y_prediction, average='weighted')

        if kernel=='linear':
            feature_importances = svm_classifier.coef_

        stat_all[i,0] = accuracy
        stat_all[i,1] = f1

        # add confusion matrix
        # df_confusion = pd.crosstab(y_test, y_prediction)
        np_confusion_all[i,:,:] = confusion_matrix(y_test, y_prediction)
        i = i +1

    # calculate the mean accuracy and confusion matrix
    accuracy_mean = np.mean(stat_all,axis=0)[0]
    f1_mean = np.mean(stat_all,axis=0)[1]

    np_confusion_mean = np.mean(np_confusion_all,axis=0)
    df_confusion_mean = pd.DataFrame(data = np_confusion_mean, columns = list(np.arange(1,np.unique(y).shape[0]+1)), index =  list(np.arange(1,np.unique(y).shape[0]+1)))

    # print ('mean accuracy: %s'%(accuracy_mean))

    return accuracy_mean, f1_mean, df_confusion_mean


def svm_classify_split(X, y):
    """
    define a function to run multiclass classification
    :param X: features set: 1600 * 7000; datatype np
    :param y: class label: 1600 * 1; datatype np
    :param kernel: the kernel of svm
    :return: classification accuracy, f1, confusion matrix cm
    """
    # create a svm with Polynomial kernel

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, train_size=0.80, test_size=0.20, random_state=101)

    # svm_classifier = svm.SVC(kernel='linear', degree=3, C=5, class_weight='balanced').fit(X_train, y_train)
    svm_classifier = svm.SVC(kernel='linear', degree=3, C=5)

    svm_classifier.fit(X_train, y_train)

    # test the classifier using the test data set
    y_prediction = svm_classifier.predict(X_test)

    # calculate the accuracy and f1 scores
    accuracy = accuracy_score(y_test, y_prediction)
    f1 = f1_score(y_test, y_prediction, average='weighted')
    feature_importances = svm_classifier.coef_


    print ('accuracy: %s'%(accuracy))

    return accuracy, f1, feature_importances

def svm_classify_top_N_features(data_features_all, labels,importances,top_N_features, data_operations):
    """

    :param data_features_all: all the features, shape: N (sample) * features
    :param labels: y labels
    :param importances: the feature importance
    :param top_N_features:
    :param data_operations:
    :return: data_operations_part: the operations of the top N features
    """

    # Sort the features based on feature importance and then select the top N to train the model
    importances = importances[0,:]
    index_features_sort = np.argsort(importances)

    # choose the top_N_features
    data_features_part = data_features_all[:, index_features_sort[-top_N_features:]]

    # use the top_N_features to classify
    X_train_part, X_test_part, y_train_part, y_test_part = model_selection.train_test_split(data_features_part, labels, train_size=0.80, test_size=0.20, random_state=101)

    clf = svm.SVC(kernel='linear', degree=3, C=5)
    clf.fit(X_train_part, y_train_part)
    y_pred_part = clf.predict(X_test_part)

    # find the feature importance
    feature_importances = clf.coef_

    # calculate the accuracy and f1 scores
    accuracy_part = accuracy_score(y_test_part, y_pred_part)
    f1_part = f1_score(y_test_part, y_pred_part, average='weighted')
    # print('use part %s features' % (top_N_features))
    # print(accuracy_part, f1_part)

    # get the id of the top_N_features
    features_top_N_ids = index_features_sort[-top_N_features:]

    # get the operations of these top features
    data_operations_part = data_operations.loc[list(features_top_N_ids)]

    data_operations_part['feature_importances'] = importances[index_features_sort[-top_N_features:]]

    return accuracy_part, f1_part, data_operations_part

# path_base = '/home/xiuyi/Data/Task_Gradient/Nifti/fMRIPrep_cifti_nifti_xcp_abcd/smooth_6_filter_min_0.01_0.08_36P_cifti/xcp_abcd_hctsa_double/sub-01R5619/ses-01/func'
# filename_features_mat = 'sub-01R5619_ses-01_task-FeatureMatching_space-fsLR_den-91k_desc-residual_smooth_bold_demean_parcel_HCTSA_N_combined.mat'
# filename_features_excel = 'sub-01R5619_ses-01_task-FeatureMatching_space-fsLR_den-91k_desc-residual_smooth_bold_demean_parcel_HCTSA_N_combined.xlsx'
#
# file_features_N = os.path.join(path_base, filename_features_mat)
# file_label = '/home/xiuyi/PycharmProjects/Qing_first_programming/labels.xlsx'
# file_operations = os.path.join(path_base,filename_features_excel)
#
# data_features_all, labels, data_operations = load_normalized_feature_vectors(file_features_N,file_label,file_operations)
#
# feature_names = data_operations['Name']
#
# accuracy, f1, importances = svm_classify_split(data_features_all, labels)
#
# importances_mean = np.mean(importances,axis=0)
# top_N_features = 40
# accuracy_part, f1_part, data_operations_part  = svm_classify_top_N_features(data_features_all, labels,importances_mean,top_N_features, data_operations)
#
#
# data_operations_part.to_excel(os.path.join('/home/xiuyi/PycharmProjects/Qing_first_programming','top_40_features_svm.xlsx'))
#




# # example to use it
# # step 1: prepare the feature vectors and labels
# # get the path of the current code
#
# path_curr = os.getcwd()
#
# file_data = os.path.join(path_curr,'data_features.mat')
# file_label = os.path.join(path_curr,'labels.xlsx')
#
# #%% step 1: load the data
# data, labels = load_normalized_feature_vectors(file_data,file_label)
#
# #%% step 2: run the classification using all the features
# accuracy_all_features, f1_all_features, cm_all_features = svm_classify_kfold(data, labels, kernel='linear')
#
# # todo save the confusion matrix - cm_all_features as an excel file
#
# #%% step 3: run the classification using part of the features
# data_part = np.array(choose_part_features(data, feature_part_num=4000))
# accuracy_part_features, f1_part_features, cm_part_features = svm_classify_kfold(data_part, labels,kernel='linear')
#
# # todo save the confusion matrix - cm_part_features as an excel file
#
# #%% step 4: check the confusion matrices have high correlation when using all the features or part of the features
# cm_all_features_vector = cm_all_features.values.flatten()
# cm_part_features_vector = cm_part_features.values.flatten()
#
# r, p = pearsonr(cm_all_features_vector,cm_part_features_vector)
#
# print (r, p )
# print ('well done')

