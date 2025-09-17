local=True
import torch
import pandas as pd
import json


class args():
    #Directory setting
    data_dir = './data/'                   # original train/test data and data dictionary should be putted here
    model_dir= './models/'                 # trained model would be saved here
    data_process_dir='./data_process/'     # processed data after basic feature engineering would be saved here
    encoder_info_dir='./encoder_info/'     # some encode-related file whould be saved here during data process
    submission_dir='./submission/'          # prediction result would be saved here

    # Data file name
    train_file = 'train.csv'                   # set None if don't train, otherwilse the file shoud be put in data_dir
    test_file =  'test.csv'                   # set None if don't reference, otherwilse the file should be put in data_dir
    prediction_result_file = 'submission.csv'  # it would be saved in submission_dir
    
    # CV stratage params for regessor and classifier training
    fold_n=10                            # CV split fold number
    seed=888                             # CV split random seed
    
    # CV stratage params form post param search
    post_CV_fold_n=5        # CV split fold number for searching post params
    post_CV_seed=888        # CV split seeds for searching post params
    
    # number of nearest neighbour nodes selected for gnn classifier and rankgnn classifier
    n_neighbors_gnn=50
    n_neighbors_rankgnn=25

    # models to be trained
    # classifier optional： ['xgboost','catboost','lightgbm','tabm','nn','gnn','ranktabm','ranknn','rankgnn']
    # regressor optional:   ['xgboost','catboost','lightgbm']
    classifiers = ['xgboost','catboost','lightgbm','tabm','nn','gnn','ranktabm','ranknn','rankgnn']
    regressors = ['xgboost','catboost','lightgbm']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    
    
    #####################################################################################################
    #Do not change the code below if not necessray
    #####################################################################################################
    if train_file is not None:
        data_train = data_dir+train_file
        sample_num_train = len(pd.read_csv(data_train))
    if test_file is not None:
        data_test = data_dir+test_file
        sample_num_test = len(pd.read_csv(data_test))

    if data_dir[-1]!='/':
        data_dir+='/'
    if model_dir[-1]!='/':
        model_dir+='/'
    if data_process_dir[-1]!='/':
        data_process_dir+='/'
    if encoder_info_dir[-1]!='/':
        encoder_info_dir+='/'
    if submission_dir[-1]!='/':
        submission_dir+='/'
