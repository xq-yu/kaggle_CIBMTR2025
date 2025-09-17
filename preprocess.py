import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import pickle
import torch
import warnings
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data
from config import args

warnings.filterwarnings('ignore')

def cat2num(df):
    '''
    input:
        df: DataFrame of features
    output:
        df: transformed DataFrame 
        
        map some category features into continual features
    '''

    df['conditioning_intensity_trans2num'] = df['conditioning_intensity'].map({
    'NMA': 1, 
    'RIC': 2,
    'MAC': 3,
    'TBD': None,
    'No drugs reported': None,
    'N/A, F(pre-TED) not submitted': None}).astype('float')
    
    df['tbi_status_trans2num'] = df['tbi_status'].map({
    'No TBI': 0, 
    'TBI +- Other, <=cGy': 1,
    'TBI +- Other, -cGy, fractionated': 2,
    'TBI + Cy +- Other': 3,
    'TBI +- Other, -cGy, single': 4,
    'TBI +- Other, >cGy': 5,
    'TBI +- Other, unknown dose': None}).astype('float')
    
    df['dri_score_trans2num'] = df['dri_score'].map({
    'Low': 1, 
    'Intermediate': 2,
    'Intermediate - TED AML case <missing cytogenetics': 3,
    'High': 4,
    'High - TED AML case <missing cytogenetics': 5,
    'Very High': 6,
    'N/A - pediatric': -3,
    'N/A - non-malignant indication': -1,
    'TBD cytogenetics': -2,
    'N/A - disease not classifiable': -4,
    'Missing disease status': 0}).astype('float')
    
    df['cyto_score_trans2num'] = df['cyto_score'].map({
    'Poor': 4,
    'Normal': 3,
    'Intermediate': 2,
    'Favorable': 1,
    'TBD': -1,
    'Other': -2,
    'Not tested': None}).astype('float')
    
    df['cyto_score_detail_trans2num'] = df['cyto_score_detail'].map({
    'Poor': 3, 
    'Intermediate': 2,
    'Favorable': 1,
    'TBD': -1,
    'Not tested': None}).astype('float')

    return df


def recalculate_hla(df):
    '''
    recalculate hla
    '''
    #df['hla_nmdp_6'] = df["hla_match_a_low"].fillna(0) + df["hla_match_b_low"].fillna(0) + df["hla_match_drb1_high"].fillna(0)
    df['hla_low_res_6'] = df["hla_match_a_low"].fillna(0) + df["hla_match_b_low"].fillna(0) + df["hla_match_drb1_low"].fillna(0)
    df['hla_high_res_6'] = df["hla_match_a_high"].fillna(0) + df["hla_match_b_high"].fillna(0) + df["hla_match_drb1_high"].fillna(0)
    df['hla_low_res_8'] = df["hla_match_a_low"].fillna(0) + df["hla_match_b_low"].fillna(0) + df["hla_match_c_low"].fillna(0) + df["hla_match_drb1_low"].fillna(0)  
    df['hla_high_res_8'] = df["hla_match_a_high"].fillna(0) + df["hla_match_b_high"].fillna(0) + df["hla_match_c_high"].fillna(0) + df["hla_match_drb1_high"].fillna(0)  
    df['hla_low_res_10'] = df["hla_match_a_low"].fillna(0) + df["hla_match_b_low"].fillna(0) + df["hla_match_c_low"].fillna(0) + df["hla_match_drb1_low"].fillna(0) + df["hla_match_dqb1_low"].fillna(0)  
    df['hla_high_res_10'] = df["hla_match_a_high"].fillna(0) + df["hla_match_b_high"].fillna(0) + df["hla_match_c_high"].fillna(0) + df["hla_match_drb1_high"].fillna(0) + df["hla_match_dqb1_high"].fillna(0)  
    return df


def feature_engineering(data_file,train):
    data = pd.read_csv(data_file)
    data_dict = pd.read_csv(args.data_dir+'data_dictionary.csv')
    feature_cat = list(data_dict.loc[data_dict.type=='Categorical','variable'])
    feature_cat = [x for x in feature_cat if x  not in ('efs','efs_time')]
    feature_value = list(data_dict.loc[data_dict.type=='Numerical','variable'])    
    feature_value = [x for x in feature_value if x  not in ('efs','efs_time')]

    # copy continual features as category features
    for col in feature_value:
        if col not in ['donor_age','age_at_hct']:
            data[col+'_trans2cat'] = data[col].copy().astype('str')
            feature_cat.append(col+'_trans2cat')
    
    features = data[feature_cat].fillna('-1')
    for col in features.columns:
        features[col]=features[col].astype('str')

    features_new = data[feature_value]#.fillna(-1)
    features = pd.concat([features,features_new],axis=1)
    
    with open(args.encoder_info_dir+'one_hot_encoder.pkl','rb') as f:
        one_hot_encoder = pickle.load(f) 
    
    # one-hot
    tmp = one_hot_encoder['features']
    features_new = one_hot_encoder['model'].transform(data[tmp]) 
    features_new=pd.DataFrame(features_new)
    features_new.columns = ['onehot_'+str(x) for x in range(len(features_new.columns))]
    features = pd.concat([features,features_new],axis=1)
    feature_onehot = list(features_new.columns)
    

    if train==True:
        return {'X':features,
                'efs':data.efs,
                'efs_time':data.efs_time,
                'feature_cat':feature_cat,
                'feature_value':feature_value,
                'feature_onehot':feature_onehot,
                'ID':data.ID,
                'race_group':data.race_group}
    else:
        return {'X':features,
                'feature_cat':feature_cat,
                'feature_value':feature_value,
                'feature_onehot':feature_onehot,
                'ID':data.ID,
                'race_group':data.race_group}


def first_data_process(data_file,train=False):
    if train:
        data = pd.read_csv(data_file)
        data_dict = pd.read_csv(args.data_dir+'data_dictionary.csv')

        # one-hot encoder
        one_hot_encoder = OneHotEncoder(sparse_output=False)
        one_hot_features = list(data_dict.loc[data_dict.type=='Categorical','variable'])
        one_hot_features = [x for x in one_hot_features if x not in ('efs','efs_time')]
        one_hot_encoder.fit(data[one_hot_features])
        one_hot_encoder_info  ={'features':one_hot_features,'model':one_hot_encoder}
        with open(args.encoder_info_dir+'one_hot_encoder.pkl','wb') as f:
            pickle.dump(one_hot_encoder_info, f)

        # features engineering
        data_train = feature_engineering(args.data_train,train=True)
        with open(args.data_process_dir+'data_train.pkl','wb') as f:
            pickle.dump(data_train,f)
        ##################features describe######################)
        print('Categary Feature number:',len(data_train['feature_cat']))
        print('numerical Feature number:',len(data_train['feature_value']))
        print('One-hot Feature number:',len(data_train['feature_onehot']))
    else:
        data_test = feature_engineering(args.data_test,train=False)
        with open(args.data_process_dir+'data_test.pkl','wb') as f:
            pickle.dump(data_test,f)



def create_edge(X,n_neighbors):
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(X)  # +1 包括自身
    distances, indices = nbrs.kneighbors(X)
    edge_index = []
    edge_weight = []
    for i, (idx,dist) in enumerate(zip(indices,distances)):
        for j in idx[1:]:  # 跳过自身
            edge_index.append([i, j])
            edge_index.append([j, i])
            edge_weight.append(1/(1+dist))
            edge_weight.append(1/(1+dist))
    return np.array(edge_index).T,np.array(edge_weight)


def create_graph(data,n_neighbors,idx=None):
    n_neighbors = min(n_neighbors,len(data['X'])-1)
    if idx is None:
        idx = list(range(len(data['X'])))

    X_cat = data['X'].iloc[idx,:][data['feature_cat']]
    X_cont = data['X'].iloc[idx,:][data['feature_value']]
    edge_index,edge_weight = create_edge(data['X'].iloc[idx,:][data['feature_value']+data['feature_onehot']],n_neighbors) 
    
    if 'efs' in data:
        y = (data['efs']==0)[idx].astype('int')
    else:
        y = np.zeros(len(data['X'])) 
    graph_data = Data(x_cat=torch.tensor(np.array(X_cat), dtype=torch.long), 
                    x_cont=torch.tensor(np.array(X_cont), dtype=torch.float32), 
                    edge_index=torch.tensor(edge_index, dtype=torch.long),
                    edge_weight=torch.tensor(edge_weight, dtype=torch.float32),
                    y=torch.tensor(np.array(y), dtype=torch.float32)
                    ).to(args.device)
    return graph_data
