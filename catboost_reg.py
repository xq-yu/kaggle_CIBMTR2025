
import pickle
from config import args
from sklearn.model_selection import StratifiedKFold
from catboost import CatBoostRegressor
from metric import fast_concordance_index
from lifelines.utils import concordance_index
import numpy as np

def cat_reg_seconde_process(data_file):
    """
    Data process for catboost regressor
    """
    with open(data_file,'rb') as f:
        data = pickle.load(f)
    for col in data['feature_cat']:
        data['X'][col] = data['X'][col].astype('category')
    input_columns = data['feature_cat']+data['feature_value']+data['feature_onehot']#+data['feature_other']#+data['feature_labelencode']
    data['X'] = data['X'][input_columns]
    data['X'] = data['X'].drop(['year_hct_trans2cat','hla_match_a_high_trans2cat','hla_match_b_low_trans2cat','comorbidity_score_trans2cat'],axis=1)
    data['feature_cat'] = [col for col in data['feature_cat'] if col not in ['year_hct_trans2cat','hla_match_a_high_trans2cat','hla_match_b_low_trans2cat','comorbidity_score_trans2cat']]

    categories = {}
    for col in data['X'].columns:
        if data['X'][col].dtype=='category':
            categories[col] = data['X'][col].cat.categories
    return data,categories

def cat_reg_train(seed,fold_n):
    """
    Catboost regressor training
    """
    data,categories = cat_reg_seconde_process(args.data_process_dir+'data_train.pkl')
    with open(args.encoder_info_dir+'./categories_cat_reg.pkl','wb') as f:
        pickle.dump(categories,f)
    model_param = {#'loss_function':'Cox',
      'eval_metric':'MAE',
      'cat_features': data['feature_cat']+['efs'],
      #'early_stopping_rounds': 100,
      'verbose': 1000,
      'random_seed': 43,
      'depth':6, 
      'learning_rate':0.02,
      'n_estimators':20000,
      'subsample':0.66,
      'colsample_bylevel':0.8,
      'l2_leaf_reg':1
      #'thread_count':11
     }
    cat_reg = CatBoostRegressor(**model_param)

    # create regress target
    efs_time_norm = data['efs_time'].copy()
    efs_time_norm[data['efs']==1] = data['efs_time'][data['efs']==1].rank()/sum(data['efs']==1)
    efs_time_norm[data['efs']==0] = data['efs_time'][data['efs']==0].rank()/sum(data['efs']==0)
    data['efs_time_norm'] = efs_time_norm

    # create sample weight
    sample_weight = np.zeros(len(data['efs']))
    sample_weight[data['efs']==1] = 0.6
    sample_weight[data['efs']==0] = 0.4
    data['sample_weight'] = sample_weight

    sample_num = len(data['X'])
    reg_prediction = np.zeros(sample_num)

    skf = StratifiedKFold(n_splits=fold_n, shuffle=True, random_state=seed)
    y_combine = data['efs'].astype('str')+'|'+data['X']['race_group'].astype('str')
    for i,(train_index,eval_index) in enumerate(skf.split(data['X'],y_combine)):
        print('############fold',i)
        efs_train = data['efs'].iloc[train_index]
        efs_eval = data['efs'].iloc[eval_index]

        X_train_reg = data['X'].iloc[train_index,:]
        X_eval_reg =  data['X'].iloc[eval_index,:]
        # add efs as an extral feature
        X_train_reg['efs'] = data['efs'].iloc[train_index].astype('int')  
        X_eval_reg['efs'] = data['efs'].iloc[eval_index].astype('int')   
        X_train_reg['efs'] = X_train_reg['efs'].astype('category').cat.set_categories([0,1])
        X_eval_reg['efs'] = X_eval_reg['efs'].astype('category').cat.set_categories([0,1])
        
        y_train_reg = data['efs_time_norm'].iloc[train_index]
        y_eval_reg = data['efs_time_norm'].iloc[eval_index]

        eval_set = [(X_train_reg, y_train_reg),\
                    (X_eval_reg[efs_eval==1], y_eval_reg[efs_eval==1])]    # only focus on the performance where efs_eval==1 when using early stop
        
        cat_reg.fit(eval_set[0][0], eval_set[0][1], 
              eval_set=eval_set[1:2], 
              use_best_model=False,
              sample_weight = data['sample_weight'][train_index]
             )
        
        # OOF predict
        y_hat_reg = cat_reg.predict(X_eval_reg) 
        reg_prediction[eval_index]=y_hat_reg

        # save model
        model_info = {'train_index':train_index,'eval_index':eval_index,'model':cat_reg}
        with open(args.model_dir+'catboost_reg_fold%s.pkl'%(i),'wb') as f:
            pickle.dump(model_info,f)

    efs_index_1 = data['efs']==1
    c_index_overall = concordance_index(data['efs_time'][efs_index_1],reg_prediction[efs_index_1],data['efs'][efs_index_1])
    print('Over All OOF REG_C-index_overall where efs==1:',c_index_overall)
    return c_index_overall
if __name__=='main':
    cat_reg_train(args.seed,args.fold_n)