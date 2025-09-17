
import pickle
from config import args
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier,XGBRegressor
from sklearn.metrics import roc_auc_score
import numpy as np
from metric import fast_concordance_index
from lifelines.utils import concordance_index
from preprocess import cat2num,recalculate_hla

def xgb_reg_seconde_process(data_file):
    with open(data_file,'rb') as f:
        data = pickle.load(f)
    for col in data['feature_cat']+data['feature_onehot']:
        data['X'][col] = data['X'][col].astype('category')
    input_columns = data['feature_cat']+data['feature_value']+data['feature_onehot']#+data['feature_other']#+data['feature_labelencode']
    data['X'] = data['X'][input_columns]
    data['X'] = data['X'].drop(['year_hct_trans2cat','hla_match_a_high_trans2cat','hla_match_b_low_trans2cat','comorbidity_score_trans2cat'],axis=1)
    data['X'] = cat2num(data['X'])
    data['X'] = recalculate_hla(data['X'])
    categories = {}
    for col in data['X'].columns:
        if data['X'][col].dtype=='category':
            categories[col] = data['X'][col].cat.categories
    return data,categories

def xgb_reg_train(seed,fold_n):
    data,categories = xgb_reg_seconde_process(args.data_process_dir+'data_train.pkl')
    with open(args.encoder_info_dir+'./categories_xgb_reg.pkl','wb') as f:
        pickle.dump(categories,f)

    model_param = {
            'n_estimators': 10000,
            'learning_rate': 0.01,
            'colsample_bytree':0.8,
            'subsample':0.8,
            #'early_stopping_rounds':100,
            'lambda':1,
            'alpha':0,
            'gamma':0,
            'eval_metric': 'mae',
            'max_depth':6,
            'verbose': -1,
            'seed': 42,
            'enable_categorical':True,
            #'n_jobs':8,
            'tree_method':'gpu_hist',
            'max_cat_to_onehot':10
        }
    xgb_reg = XGBRegressor(**model_param)

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
                    (X_eval_reg[efs_eval==1], y_eval_reg[efs_eval==1])]    # focus the metric where esf==1 when using early stop
        
        xgb_reg.fit(eval_set[0][0], eval_set[0][1], 
                eval_set=eval_set[0:2], 
                sample_weight = data['sample_weight'][train_index],
                verbose=1000
                )
        
        y_hat_reg = xgb_reg.predict(X_eval_reg) 
        reg_prediction[eval_index]=y_hat_reg

        model_info = {'train_index':train_index,'eval_index':eval_index,'model':xgb_reg}
        with open(args.model_dir+'xgboost_reg_fold%s.pkl'%(i),'wb') as f:
            pickle.dump(model_info,f)

    efs_index_1 = data['efs']==1
    c_index_overall = concordance_index(data['efs_time'][efs_index_1],reg_prediction[efs_index_1],data['efs'][efs_index_1])
    print('Over All OOF REG_C-index_overall where efs==1:',c_index_overall)
    return c_index_overall


if __name__=='__main__':
    xgb_reg_train(args.seed,args.fold_n)