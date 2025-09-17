from lightgbm_cls import lgb_cls_seconde_process
from catboost_cls import cat_cls_seconde_process
from xgboost_cls import xgb_cls_seconde_process
from nn_cls import nn_cls_seconde_process
from tabm_cls import tabm_cls_seconde_process
from gnn_cls import gnn_cls_seconde_process
from ranknn_cls import ranknn_cls_seconde_process
from ranktabm_cls import ranktabm_cls_seconde_process
from rankgnn_cls import rankgnn_cls_seconde_process
from lightgbm_reg import lgb_reg_seconde_process
from catboost_reg import cat_reg_seconde_process
from xgboost_reg import xgb_reg_seconde_process
from preprocess import create_graph
import torch
import numpy as np
import pandas as pd
import pickle
from config import args
from preprocess import first_data_process

from tabm_cls import TabM_cls
from nn_cls import LitNN_cls,NN_cls,CatEmbeddings_cls_nn
from gnn_cls import graphsage,CatEmbeddings_cls_gnn
from ranktabm_cls import RankTabM_cls
from ranknn_cls import RankLitNN_cls,RankNN_cls,CatEmbeddings_cls_ranknn
from rankgnn_cls import Rankgraphsage,CatEmbeddings_cls_rankgnn


def cls_predict(model_nm,fold_n,test=False):
    '''
    input:
        model_nm: xgboost/catboost/lightgbm/nn/tabm/gnn/ranknn/ranktabm/rankgnn
        fold_n: CV fold number when training the model
        test: 
            True:inferece in the test dataset  with models and the blend weth average
            False: inferece the oof result in the train dataset
    output:
        predictied result of test data or oof
    '''

    # data process
    if test:
        data_file = args.data_process_dir+'data_test.pkl'
    else:
        data_file = args.data_process_dir+'data_train.pkl'

    # data process for specific model
    if model_nm=='lightgbm':
        data,_ = lgb_cls_seconde_process(data_file)
    elif model_nm=='catboost':
        data,_ = cat_cls_seconde_process(data_file)
    elif model_nm=='xgboost':
        data,_ = xgb_cls_seconde_process(data_file)
    elif model_nm=='nn':
        with open(args.encoder_info_dir+'./encoders_nn_cls.pkl','rb') as f:
            enconders = pickle.load(f)
        data,_,_,_ = nn_cls_seconde_process(data_file,enconders['label_encoders'],enconders['imputers'],enconders['scalers'])
    elif model_nm=='tabm':
        with open(args.encoder_info_dir+'./encoders_tabm_cls.pkl','rb') as f:
            enconders = pickle.load(f)
        data,_,_,_ = tabm_cls_seconde_process(data_file,enconders['label_encoders'],enconders['imputers'],enconders['scalers'])
    elif model_nm=='gnn':
        with open(args.encoder_info_dir+'./encoders_gnn_cls.pkl','rb') as f:
            enconders = pickle.load(f)
        data,_,_,_ = gnn_cls_seconde_process(data_file,enconders['label_encoders'],enconders['imputers'],enconders['scalers'])
    elif model_nm=='ranknn':
        with open(args.encoder_info_dir+'./encoders_ranknn_cls.pkl','rb') as f:
            enconders = pickle.load(f)
        data,_,_,_ = ranknn_cls_seconde_process(data_file,enconders['label_encoders'],enconders['imputers'],enconders['scalers'])
    elif model_nm=='ranktabm':
        with open(args.encoder_info_dir+'./encoders_ranktabm_cls.pkl','rb') as f:
            enconders = pickle.load(f)
        data,_,_,_ = ranktabm_cls_seconde_process(data_file,enconders['label_encoders'],enconders['imputers'],enconders['scalers'])
    elif model_nm=='rankgnn':
        with open(args.encoder_info_dir+'./encoders_rankgnn_cls.pkl','rb') as f:
            enconders = pickle.load(f)
        data,_,_,_ = rankgnn_cls_seconde_process(data_file,enconders['label_encoders'],enconders['imputers'],enconders['scalers'])
  

    categories_file = {'xgboost': args.encoder_info_dir+'categories_xgb_cls.pkl',
                       'catboost': args.encoder_info_dir+'categories_cat_cls.pkl',
                       'lightgbm': args.encoder_info_dir+'categories_lgb_cls.pkl'}
    if model_nm in categories_file.keys():
        with open(categories_file[model_nm],'rb') as f:
            categories = pickle.load(f)
        for col in data['X'].columns:
            if data['X'][col].dtype=='category':
                data['X'][col] = data['X'][col].cat.set_categories(categories[col]) 


    output = np.zeros(len(data['X']))
    for i in range(fold_n):
        # load model
        if model_nm in ('xgboost','lightgbm','catboost'):
            with open(args.model_dir+f'{model_nm}_cls_fold{i}.pkl','rb') as f:
                model_info = pickle.load(f) 
        else:
            model_info = torch.load(args.model_dir+f'{model_nm}_cls_fold{i}.pth')
            model_info['model'] =  model_info['model'].to(args.device)

        model = model_info['model']

        if test:
            if model_nm in ('gnn','rankgnn'):
                # create graph data
                n_neighbors = args.n_neighbors_gnn if model_nm=='gnn' else args.n_neighbors_rankgnn
                graph_data = create_graph(data,n_neighbors=n_neighbors)
                y_hat = model.predict_proba(graph_data)[:,1]
            else:
                y_hat = model.predict_proba(data['X'])[:,1]
            output+=y_hat/fold_n
        else:
            # predict OOF
            eval_index = model_info['eval_index']
            if model_nm in ('gnn','rankgnn'):
                n_neighbors = args.n_neighbors_gnn if model_nm=='gnn' else args.n_neighbors_rankgnn
                graph_data = create_graph(data,n_neighbors=n_neighbors,idx=eval_index)
                y_hat = model.predict_proba(graph_data)[:,1]
            else:
                y_hat = model.predict_proba(data['X'].loc[eval_index])[:,1]            
            output[eval_index] = y_hat
    return output


def reg_predict(model_nm,fold_n,test=False):
    '''
    input:
        model_nm: xgboost/catboost/lightgbm
        fold_n: CV fold number when the model was training
        test: 
            True:inferece in the test dataset
            False: inferece the oof result in the train dataset
    output:
        predictied result of test data or oof
    '''

    # data process
    if test:
        data_file = args.data_process_dir+'data_test.pkl'
    else:
        data_file = args.data_process_dir+'data_train.pkl'
    if model_nm=='lightgbm':    
        data,_ = lgb_reg_seconde_process(data_file)
    if model_nm=='xgboost':
        data,_ = xgb_reg_seconde_process(data_file)
    if model_nm=='catboost':
        data,_ = cat_reg_seconde_process(data_file)
 
    categories_file = {'xgboost': args.encoder_info_dir+'categories_xgb_reg.pkl',
                       'catboost': args.encoder_info_dir+'categories_cat_reg.pkl',
                       'lightgbm': args.encoder_info_dir+'categories_lgb_reg.pkl'}
    if model_nm in categories_file.keys():
        with open(categories_file[model_nm],'rb') as f:
            categories = pickle.load(f)
        for col in data['X'].columns:
            if data['X'][col].dtype=='category':
                data['X'][col] = data['X'][col].cat.set_categories(categories[col]) 

    data['X']['efs']=1
    data['X']['efs'] = data['X']['efs'].astype('category').cat.set_categories([0,1])


    output = np.zeros(len(data['X']))
    for i in range(fold_n):
        # load model
        if model_nm in ('xgboost','lightgbm','catboost'):
            with open(args.model_dir+f'{model_nm}_reg_fold{i}.pkl','rb') as f:
                model_info = pickle.load(f)
        model = model_info['model']

        if test:
            y_hat = model.predict(data['X'])
            output+=y_hat/fold_n
        else:
            eval_index = model_info['eval_index']
            output[eval_index] = model.predict(data['X'].loc[eval_index])
    return output


def merge_fun(Y_HAT_REG,Y_HAT_CLS,a=2.96,b=1.77,c=0.52):
    y_fun = (Y_HAT_REG>0)*c*np.abs(Y_HAT_REG)**b
    x_fun = (Y_HAT_CLS>0)*np.abs(Y_HAT_CLS)**a
    res = (1-y_fun)*x_fun+y_fun
    # **make sure there is enough samples**
    res = pd.Series(res).rank()/len(res)
    return res


def model_merge(merge_params):
    cls_models = set([k.split('|')[0] for k in merge_params.keys()])
    reg_models = set([k.split('|')[1] for k in merge_params.keys()])
    y_hat_cls = {}
    y_hat_reg = {}

    print('######classifier inference...')
    for model_nm in cls_models:
        print(model_nm)
        y_hat_cls[model_nm] = cls_predict(model_nm=model_nm,fold_n=args.fold_n,test=True)
    
    print('######regressor inference...')
    for model_nm in reg_models:
        print(model_nm)
        y_hat_reg[model_nm] = reg_predict(model_nm=model_nm,fold_n=args.fold_n,test=True)

    
    print('######results merge...')
    results = {}
    for comb in merge_params.keys():
        print(comb)
        cls_model,reg_model = comb.split('|')
        # mean of merged result by merge-params from different folds.
        result_avg = []
        for fold in range(args.post_CV_fold_n):
            result_avg.append(merge_fun(y_hat_reg[reg_model],y_hat_cls[cls_model],**merge_params[comb][fold]))
        result_avg = np.mean(np.array(result_avg).T,axis=1)
        results[comb] = result_avg
    return results


def results_ensemble(results,weights):
    '''
    Sum the results with weighted average
    '''
    weight_sum = sum(weights.values())
    weights = {k:v/weight_sum for k,v in weights.items()}

    predict = []
    for model_nm in weights.keys():
        predict.append(weights[model_nm]*results[model_nm])
    return np.sum(np.array(predict).T,axis=1)


if __name__=='__main__':
    first_data_process(args.data_test,train=False)

    print('model merge...')
    with open(args.model_dir+'merge_params.pkl','rb')  as f:
        merge_params = pickle.load(f)
    merge_results = model_merge(merge_params)

    with open(args.model_dir+'ensemble_weights.pkl','rb') as f:
        ensemble_weights = pickle.load(f)

    # mean of ensemble result by weights from different folds.
    print('#####ensemble the merged results of classifiers and regressors....')
    y_hat = []
    for i in range(args.post_CV_fold_n):
        y_hat.append(results_ensemble(merge_results,ensemble_weights[i]))
    y_hat = np.mean(np.array(y_hat).T,axis=1)

    # save the result
    data_test = pd.read_csv(args.data_test)
    pred = pd.DataFrame({'ID':data_test['ID'],'prediction':1-y_hat})
    pred.to_csv(args.submission_dir+args.prediction_result_file,index=False)

    print('#####finished')