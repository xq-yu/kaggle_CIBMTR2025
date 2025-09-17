from lightgbm_cls import lgb_cls_train
from catboost_cls import cat_cls_train
from xgboost_cls import xgb_cls_train
from lightgbm_reg import lgb_reg_train
from catboost_reg import cat_reg_train
from xgboost_reg import xgb_reg_train
from nn_cls import nn_cls_train
from tabm_cls import tabm_cls_train
from gnn_cls import gnn_cls_train
from ranknn_cls import ranknn_cls_train
from ranktabm_cls import ranktabm_cls_train
from rankgnn_cls import rankgnn_cls_train

from post_param_search import search_best_merge_params,search_best_ensemble_weights

from preprocess import first_data_process
from config import args

if __name__=='__main__':
    # data preprocess
    first_data_process(args.data_train,train=True)

    
    # classifier training
    print('#####start classifier training')
    for model_nm in args.classifiers:
        print('###############')
        print(model_nm)
        print('###############')
        if model_nm=='lightgbm':
            lgb_cls_train(args.seed,args.fold_n)
        elif model_nm=='catboost':
            cat_cls_train(args.seed,args.fold_n)
        elif model_nm=='xgboost':
            xgb_cls_train(args.seed,args.fold_n)
        elif model_nm=='nn':
            nn_cls_train(args.seed,args.fold_n)
        elif model_nm=='tabm':        
            tabm_cls_train(args.seed,args.fold_n)
        elif model_nm=='gnn':    
            gnn_cls_train(args.seed,args.fold_n)
        elif model_nm=='ranknn':
            ranknn_cls_train(args.seed,args.fold_n)
        elif model_nm=='ranktabm':
            ranktabm_cls_train(args.seed,args.fold_n)
        elif model_nm=='rankgnn':
            rankgnn_cls_train(args.seed,args.fold_n)
        print('\n')

    # regressor training
    print('#####start regressor training')
    for model_nm in args.regressors:
        print('###############')
        print(model_nm)
        print('###############')
        if model_nm=='lightgbm':
            lgb_reg_train(args.seed,args.fold_n)
        elif model_nm=='catboost':
            cat_reg_train(args.seed,args.fold_n)
        elif model_nm=='xgboost':
            xgb_reg_train(args.seed,args.fold_n)
        print('\n')
    
    # post param search
    merged_results = search_best_merge_params(args.classifiers,args.regressors)
    search_best_ensemble_weights(merged_results)

