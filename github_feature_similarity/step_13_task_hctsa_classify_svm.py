# classify the network based on the features of parcel

    from function_classification_network_svm import  load_normalized_feature_vectors, choose_part_features, svm_classify_kfold
import os
import numpy as np
import pandas as pd

dataset = 1

if dataset == 1:
    path_base_input = '/home/xiuyi/Data/Task_Gradient/Nifti/fMRIPrep_cifti_nifti_xcp_abcd/smooth_6_filter_min_0.01_0.08_36P_cifti/xcp_abcd_hctsa_double'
    path_output = '/home/xiuyi/Data/Task_Gradient/Nifti/fMRIPrep_cifti_nifti_xcp_abcd/smooth_6_filter_min_0.01_0.08_36P_cifti/xcp_abcd_hctsa_double_confusion_matrix/combined_data_classification_svm_python'
    ses_IDs = ['ses-01', 'ses-02']
    tasks = ['FeatureMatching','Association']

elif dataset == 2:
    path_base_input = '/home/xiuyi/Data/Task_Gradient_localizers/Brain_data/Nifti/fMRIPrep_cifti_nifti_xcp_abcd/smooth_6_filter_min_0.01_0.08_36P_cifti/xcp_abcd_hctsa_double'
    path_output = '/home/xiuyi/Data/Task_Gradient_localizers/Brain_data/Nifti/fMRIPrep_cifti_nifti_xcp_abcd/smooth_6_filter_min_0.01_0.08_36P_cifti/xcp_abcd_hctsa_double_confusion_matrix/combined_data_classification_svm_python'
    ses_IDs = ['ses-02', 'ses-02']
    tasks = ['Spatial', 'Maths']

os.makedirs(path_output,exist_ok=True)
sub_IDs = os.listdir(path_base_input)

sub_IDs.sort()

folder_func = 'func'
file_suffix = 'space-fsLR_den-91k_desc-residual_smooth_bold_demean_parcel_HCTSA_N_combined.mat'

file_label = '/home/xiuyi/Data/Atlas/Parcellation_Kong_2022/labels.xlsx'
feature_part_num = 4000
for i in range(len(ses_IDs)):
    ses_ID = ses_IDs[i]
    task = tasks[i]

    if task == 'Spatial':
        sub_IDs = sub_IDs[1::]

    accuracy_all = np.zeros((len(sub_IDs),2))
    np_cm_all_features = np.zeros((len(sub_IDs),17,17))
    np_cm_part_features = np.zeros((len(sub_IDs),17,17))


    for j in range(len(sub_IDs)):

        print (j)
        sub_ID = sub_IDs[j]
        filename = '%s_%s_task-%s_%s'%(sub_ID,ses_ID,task,file_suffix)

        file_data = os.path.join(path_base_input,sub_ID,ses_ID,folder_func,filename)

        # %% step 1: load the data
        data, labels = load_normalized_feature_vectors(file_data, file_label)

        # %% step 2: run the classification using all the features
        accuracy_all_features, f1_all_features, cm_all_features = svm_classify_kfold(data, labels, kernel='linear')

        # todo save the confusion matrix - cm_all_features as an excel file

        # %% step 3: run the classification using part of the features
        data_part = np.array(choose_part_features(data, feature_part_num=feature_part_num))
        accuracy_part_features, f1_part_features, cm_part_features = svm_classify_kfold(data_part, labels, kernel='linear')

        accuracy_all[j,0]=accuracy_all_features
        accuracy_all[j,1]=accuracy_part_features
        np_cm_all_features[j,:,:]=cm_all_features.values
        np_cm_part_features[j,:,:]=cm_part_features.values

    np_cm_all_mean = np.mean(np_cm_all_features,axis=0)
    np_cm_part_mean = np.mean(np_cm_part_features,axis=0)

    df_accuracy = pd.DataFrame(data = accuracy_all ,columns = ['all_features','%s_features'%(feature_part_num)] )
    df_accuracy['participants'] = sub_IDs

    df_accuracy.to_excel(os.path.join(path_output,'%s_classification_accuracy_3000.xlsx'%(task)),index=False)

    df_cm_all = pd.DataFrame(data=np_cm_all_mean, columns=cm_all_features.columns, index = cm_all_features.index)
    df_cm_part = pd.DataFrame(data=np_cm_part_mean, columns=cm_part_features.columns, index = cm_part_features.index)

    df_cm_all.to_excel(os.path.join(path_output,'%s_classification_cm_all_features.xlsx'%(task)),index=True)
    df_cm_part.to_excel(os.path.join(path_output,'%s_classification_cm_%s_features.xlsx'%(task,feature_part_num)),index=True)

    np.save(os.path.join(path_output,'%s_classification_cm_all_features.npy')%(task), np_cm_all_features)
    np.save(os.path.join(path_output,'%s_classification_cm_%s_features.npy'%(task,feature_part_num)), np_cm_part_features)


print ('well done')