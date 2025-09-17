from inference import merge_fun,results_ensemble,cls_predict,reg_predict
from metric import CIBMTR_score
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np
import optuna
from config import args
import pickle
from matplotlib import pyplot as plt
from lifelines.utils import concordance_index
from sklearn.metrics import roc_auc_score
optuna.logging.set_verbosity(optuna.logging.WARNING)

# import NN Framework
from tabm_cls import TabM_cls
from nn_cls import LitNN_cls,NN_cls,CatEmbeddings_cls_nn
from gnn_cls import graphsage,CatEmbeddings_cls_gnn
from ranktabm_cls import RankTabM_cls
from ranknn_cls import RankLitNN_cls,RankNN_cls,CatEmbeddings_cls_ranknn
from rankgnn_cls import Rankgraphsage,CatEmbeddings_cls_rankgnn


def combine_objective(trial,y_hat_reg,y_hat_cls,efs_time,efs,race_group,cls_model_nm):
    '''
    Objective funtion of optuna for search the best merge function params
    input:
        trial: optuna trial param
        y_hat_reg: regressor prediction
        y_hat_cls: classifier predction
        efs_time: efs_time of train dataset
        efs: efs of train dataset
        race_group: race group of train dataset
        cls_model_nm: classifier model name(set different search range for differen model)
    output:
        score: Stratified c-index
    '''

    if cls_model_nm in ('ranknn','rankgnn','ranktabm'):
        params = {'Y_HAT_REG':y_hat_reg,
                'Y_HAT_CLS':y_hat_cls,
                'a':trial.suggest_uniform("a", 4, 6),
                'b':trial.suggest_uniform('b',0.5,1.5),
                'c':trial.suggest_uniform('c',0,1),
                }
    else:
        params = {'Y_HAT_REG':y_hat_reg,
                'Y_HAT_CLS':y_hat_cls,
                'a':trial.suggest_uniform("a", 2, 3.5),
                'b':trial.suggest_uniform('b',0.5,1.5),
                'c':trial.suggest_uniform('c',0,1),
                }

    score,var_error,metric_list = CIBMTR_score(efs_time,merge_fun(**params),efs,race_group)
    return score


def merge_param_fit(efs,efs_time,race_group,y_hat_reg,y_hat_cls,cls_model_nm,fold_n=args.post_CV_fold_n,seed=args.post_CV_seed):
    '''
    Find the best param for model merge functiont with optuna and K-fold cross validation, given predicted cls result and reg result
    
    input:
        y_hat_reg: regressor prediction
        y_hat_cls: classifier predction
        efs_time: efs_time of train dataset
        efs: efs of train dataset
        race_group: race group of train dataset
        cls_model_nm: classifier model name(set different search range for differen model)      
    return:
        result_combine: oof merged result
        best_params: best param for merge function
    '''

    y_combine = pd.Series(efs).astype('str')+'|'+pd.Series(race_group).astype('str')
    skf = StratifiedKFold(n_splits=fold_n, shuffle=True, random_state=seed)
    result_combine = np.zeros(len(efs))
    
    best_params = []
    for i,(train_index,eval_index) in enumerate(skf.split(efs,y_combine)):
        study=optuna.create_study(direction='maximize')
        study.optimize(lambda trial: combine_objective(trial, 
                                                       y_hat_reg[train_index],
                                                       y_hat_cls[train_index],
                                                       efs_time[train_index],
                                                       efs[train_index],
                                                       race_group[train_index],
                                                       cls_model_nm
                                                       ), 
                                                       n_trials=200)
        
        result_combine[eval_index]=merge_fun(y_hat_reg[eval_index],y_hat_cls[eval_index],**study.best_params)
        best_params.append(study.best_params)
        print(f'Best merge param fold{i}:',study.best_params)
    return result_combine,best_params

def search_best_merge_params(classifiers,regressors):
    '''
    Find the best merge function params as save to merge_params.pkl with cross validation, and return the oof merge results
    input:
        classifiers: list of classifier name
        regressors: list of regressor name

    output:
        merge_params: saved to merge_params.pkl   {'cls_model|reg_model':[param_fold0,param_fold1,param_fold2,...],...}, 
        merge_result: a dict of oof merged results  {'cls_model|reg_model':y_hat,....}
    '''
    merge_params = {}
    merged_results = {}
    
    with open(args.data_process_dir+'data_train.pkl','rb') as f:
        data = pickle.load(f)
        efs = np.array(data['efs'])
        efs_time = np.array(data['efs_time'])
        race_group = np.array(data['race_group'])

    y_hat_cls = {}
    for cls_model in classifiers:
        y_hat_cls[cls_model]=cls_predict(model_nm=cls_model,fold_n=args.fold_n,test=False)
        auc = roc_auc_score(efs==0,y_hat_cls[cls_model])
        print(f'{cls_model} cls auc:',auc)
    y_hat_reg = {}
    for reg_model in regressors:
        y_hat_reg[reg_model]=reg_predict(model_nm=reg_model,fold_n=args.fold_n,test=False)
        c_index = concordance_index(efs_time[efs==1],y_hat_reg[reg_model][efs==1])
        print(f'{reg_model} reg c-index:',c_index)
            
    for cls_model in classifiers:
        for reg_model in regressors:
            print(f'search best merge param for {cls_model}|{reg_model}')
            # search the merge params and get the oof merge result 
            y_hat_merge,best_params = merge_param_fit(
                efs=efs,
                efs_time=efs_time,
                race_group=race_group,
                y_hat_reg=y_hat_reg[reg_model],
                y_hat_cls=y_hat_cls[cls_model],
                cls_model_nm=cls_model
            )  
            merge_params[f'{cls_model}|{reg_model}']=best_params
            merged_results[f'{cls_model}|{reg_model}']=y_hat_merge
   
            c_index_overall,var_error,metric_list = CIBMTR_score(efs_time,y_hat_merge,efs,race_group)
            print(f'OOF Stratified C-index of {cls_model}|{reg_model}:   ',c_index_overall)

            print('\n\n')

    with open(args.model_dir+'merge_params.pkl','wb') as f:
        pickle.dump(merge_params,f)

    return merged_results



def ensemble_objective(trial,merged_results,efs,efs_time,race_group):
    '''
    objective function for find the best weight for each model with optuna

    input:
        trial: optuna trial param
        merged_results: a dict of merged results from all combinations of defferet classifiers and regressors
  
    output:
        score: Stratified c-index
    '''
    weights = { k:trial.suggest_uniform(k, 0.1, 1) for k  in merged_results.keys()}
    ensembled_result= results_ensemble(merged_results,weights)
    score,var_error,metric_list = CIBMTR_score(efs_time,ensembled_result,efs,race_group)
    return score


def search_best_ensemble_weights(merged_results):
    '''
    Find the best weights of all merged results
    input: 
        merged_results:a dict of all merged results
    output:
        save best weights to ensemble_weights.pkl
    '''

    with open(args.data_process_dir+'data_train.pkl','rb') as f:
        data = pickle.load(f)
        efs = data['efs']
        efs_time = data['efs_time']
        race_group = data['race_group']

    y_combine = pd.Series(efs).astype('str')+'|'+pd.Series(race_group).astype('str')
    skf = StratifiedKFold(n_splits=args.post_CV_fold_n, shuffle=True, random_state=args.post_CV_seed)
    best_weights=[]
    ensembled_result = np.zeros(len(data['X']))
    for i,(train_index,eval_index) in enumerate(skf.split(efs_time,y_combine)):
        print(f'####fold{i}')
        study=optuna.create_study(direction='maximize')

        merged_results_train = {k:v[train_index] for k,v in merged_results.items()}
        study.optimize(lambda trial:ensemble_objective(trial,merged_results_train,efs_time[train_index],efs[train_index],race_group[train_index]), n_trials=800)
        best_weights.append(study.best_params)
        merged_results_eval = {k:v[eval_index] for k,v in merged_results.items()}
        ensembled_result[eval_index] = results_ensemble(merged_results_eval,study.best_params)

    with open(args.model_dir+'ensemble_weights.pkl','wb') as f:
        pickle.dump(best_weights,f)

    c_index_overall,var_error,metric_list = CIBMTR_score(efs_time,ensembled_result,efs,race_group)
    print(f'OOF Stratified C-index with weighted average ensembled:',c_index_overall)

if __name__=='__main__':
    merged_results = search_best_merge_params(args.classifiers,args.regressors)
    search_best_ensemble_weights(merged_results)
