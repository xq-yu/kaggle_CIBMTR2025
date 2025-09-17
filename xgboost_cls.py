import pickle
from config import args
from preprocess import *
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

def xgb_cls_seconde_process(data_file):
    """
    Data process for xgboost classifier
    """
    with open(data_file,'rb') as f:
        data = pickle.load(f)
    for col in data['feature_cat']:
        data['X'][col] = data['X'][col].astype('category')
    input_columns = data['feature_cat']+data['feature_value']+data['feature_onehot']
    data['X'] = data['X'][input_columns]
    data['X'] = cat2num(data['X'])
    data['X'] = recalculate_hla(data['X'])

    categories = {}
    for col in data['X'].columns:
        if data['X'][col].dtype=='category':
            categories[col] = data['X'][col].cat.categories
    return data,categories


def xgb_cls_train(seed,fold_n):
    data,categories = xgb_cls_seconde_process(args.data_process_dir+'data_train.pkl')
    with open(args.encoder_info_dir+'./categories_xgb_cls.pkl','wb') as f:
        pickle.dump(categories,f)
    data['cls_target'] = (data['efs']==0).astype('int')

    # define model
    params = {
            'n_estimators': 2500,
            'learning_rate': 0.02,
            'colsample_bytree':0.8,
            'subsample':0.8,
            #'early_stopping_rounds':100,
            'lambda':0.01,
            'alpha':0,
            'gamma':0,
            'eval_metric': 'auc',
            'max_depth':2,
            'verbose': -1,
            'seed': 42,
            'enable_categorical':True,
            'n_jobs':-1,
            #'max_cat_to_onehot':10
        }
    xgb_cls = XGBClassifier(**params)

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
        xgb_cls.fit(eval_set[0][0], eval_set[0][1], 
            eval_set=eval_set,
            verbose=1000
            )
        y_hat_cls = xgb_cls.predict_proba(X_eval_cls)[:,1]
        cls_prediction[eval_index]=y_hat_cls

        model_info = {'train_index':train_index,'eval_index':eval_index,'model':xgb_cls}
        with open(args.model_dir+'xgboost_cls_fold%s.pkl'%(i),'wb') as f:
            pickle.dump(model_info,f)

    ##########################模型评估#####################################
    print('Over All OOF CLS-AUC:',roc_auc_score(data['cls_target'],cls_prediction))

if __name__=='__main__':
    xgb_cls_train(args.seed,args.fold_n)