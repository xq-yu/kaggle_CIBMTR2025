
import pickle
from config import args
from sklearn.model_selection import StratifiedKFold
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
import numpy as np

def cat_cls_seconde_process(data_file):
    """
    Data process for catboost classifier
    """
    with open(data_file,'rb') as f:
        data = pickle.load(f)
    for col in data['feature_cat']:
        data['X'][col] = data['X'][col].astype('category')
    input_columns = data['feature_cat']+data['feature_value']+data['feature_onehot']
    data['X'] = data['X'][input_columns]
    categories = {}
    for col in data['X'].columns:
        if data['X'][col].dtype=='category':
            categories[col] = data['X'][col].cat.categories
    return data,categories

def cat_cls_train(seed,fold_n):
    """
    Catboost Classifier training
    """


    data,categories = cat_cls_seconde_process(args.data_process_dir+'data_train.pkl')
    with open(args.encoder_info_dir+'./categories_cat_cls.pkl','wb') as f:
        pickle.dump(categories,f)
    data['cls_target'] = (data['efs']==0).astype('int')

    params = {'loss_function':'Logloss',
            'eval_metric':'AUC',
            'cat_features': data['feature_cat'],
            #'early_stopping_rounds': 100,
            'verbose': 1000,
            'random_seed': 43,
            'depth':6, 
            'learning_rate':0.02,
            'n_estimators':2000,
            'l2_leaf_reg':1,
            #'thread_count':11
            'subsample':0.66,
            'colsample_bylevel':0.8,
            }
    catbst_cls = CatBoostClassifier(**params)

    sample_num = len(data['X'])
    cls_prediction = np.zeros(sample_num)

    skf = StratifiedKFold(n_splits=fold_n, shuffle=True, random_state=seed)
    y_combine = data['efs'].astype('str')+'|'+data['X']['race_group'].astype('str')
    for i,(train_index,eval_index) in enumerate(skf.split(data['X'],y_combine)):
        print('############fold',i)

        X_train_cls = data['X'].iloc[train_index,:]
        y_train_cls = data['cls_target'][train_index]
        X_eval_cls = data['X'].iloc[eval_index,:]
        y_eval_cls = data['cls_target'][eval_index]


        eval_set =[(X_train_cls, y_train_cls),\
                (X_eval_cls, y_eval_cls)]  
        catbst_cls.fit(eval_set[0][0], eval_set[0][1], 
                eval_set=eval_set[1:2], 
                use_best_model=False
                )
        y_hat_cls = catbst_cls.predict_proba(X_eval_cls)[:,1]
        cls_prediction[eval_index]=y_hat_cls

        model_info = {'train_index':train_index,'eval_index':eval_index,'model':catbst_cls}
        with open(args.model_dir+'catboost_cls_fold%s.pkl'%(i),'wb') as f:
            pickle.dump(model_info,f)

    print('Over All OOF CLS-AUC:',roc_auc_score(data['cls_target'],cls_prediction))

if __name__=='main':
    cat_cls_train(args.seed,args.fold_n)