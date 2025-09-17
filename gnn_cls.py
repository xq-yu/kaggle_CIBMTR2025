import pickle
from sklearn.model_selection import StratifiedKFold
from torch import nn
from typing import List
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.metrics import roc_auc_score
from config import args
from preprocess import create_graph
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


def gnn_cls_seconde_process(data_file,label_encoders=None,imputers=None,scalers=None):
    '''
    data process for gnn model
    input:
        data_file: first preprocessed data
        label_encoders: label_encoders created when training, if None,will be created
        imputers: imputers created when training, if None,will be created
        scalers: scalers created when training, if None,will be created
    '''

    with open(data_file,'rb') as f:
        data = pickle.load(f)
    input_columns = data['feature_cat']+data['feature_value']+data['feature_onehot']
    data['X'] = data['X'][input_columns]

    # scale | fillna | label encode
    if label_encoders is None:
        label_encoders = {col:LabelEncoder().fit(data['X'][col]) for col in data['feature_cat']}
    for col in data['feature_cat']:
        data['X'][col] = label_encoders[col].transform(data['X'][col])
    if imputers is None:
        imputers = {col:SimpleImputer(missing_values=np.nan, strategy='mean', add_indicator=False).fit(data['X'][[col]]) for col in data['feature_value']+data['feature_onehot']}
    for col in data['feature_value']+data['feature_onehot']:
        data['X'][col] = imputers[col].transform(data['X'][[col]])[:,0]
    if scalers is None:
        scalers = {col:StandardScaler().fit(data['X'][[col]]) for col in data['feature_value']+data['feature_onehot']}
    for col in data['feature_value']+data['feature_onehot']:
        data['X'][col] = scalers[col].transform(data['X'][[col]])[:,0]
    return data,label_encoders,imputers,scalers


class CatEmbeddings_cls_gnn(nn.Module):
    """
    Embedding module for the categorical dataframe.
    """
    def __init__(
        self,
        projection_dim: int,
        categorical_cardinality: List[int],
        embedding_dim: int
    ):
        super(CatEmbeddings_cls_gnn, self).__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(cardinality, embedding_dim)
            for cardinality in categorical_cardinality
        ])
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim * len(categorical_cardinality), projection_dim),
            nn.GELU(),
            nn.Linear(projection_dim, projection_dim),
            nn.GELU(),
            nn.Linear(projection_dim, projection_dim)
        )

    def forward(self, x_cat):
        """
        Apply the projection on concatened embeddings that contains all categorical features.
        """
        x_cat = [embedding(x_cat[:, i]) for i, embedding in enumerate(self.embeddings)]
        x_cat = torch.cat(x_cat, dim=1)
        return self.projection(x_cat)


class graphsage(torch.nn.Module):
    def __init__(self, 
                 continual_feature_num, 
                 hidden_channels, 
                 out_channels,
                 categorical_cardinality,
                 embedding_dim,
                 embedding_projection_dim
                 ):
        super(graphsage, self).__init__()
        self.embeddings = CatEmbeddings_cls_gnn(embedding_projection_dim, categorical_cardinality, embedding_dim)
        self.conv1 = SAGEConv(continual_feature_num+embedding_projection_dim, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, hidden_channels)
        self.conv4 = SAGEConv(hidden_channels, hidden_channels)
        self.conv5 = SAGEConv(hidden_channels, out_channels)
        self.sig = nn.Sigmoid()
        
        self.predict_shift = None  #used to fix the result shift, keep same mean prediction value for differet model

    def forward(self, x_cat,x_cont, edge_index,edge_weight=None):
        x = self.embeddings(x_cat)
        x = torch.cat([x, x_cont], dim=1)
        x = F.gelu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.gelu(self.conv2(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.gelu(self.conv3(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv5(x, edge_index)
        x = self.sig(x)
        return x[:,0]
    
    def predict_proba(self,graph_data):
        pred1= self(graph_data.x_cat,
            graph_data.x_cont,
            graph_data.edge_index,
            graph_data.edge_weight
        ).cpu().detach().numpy()

        if self.predict_shift is not None:
            pred1 = self.predict_shift*pred1
        pred0 = 1-pred1
        return np.array([pred0,pred1]).T
    
def gnn_cls_train(seed,fold_n):
    # data process
    data,label_encoders,imputers,scalers = gnn_cls_seconde_process(args.data_process_dir+'data_train.pkl')
    encoders = {'label_encoders':label_encoders,'imputers':imputers,'scalers':scalers}
    with open(args.encoder_info_dir+'encoders_gnn_cls.pkl','wb') as f:
        pickle.dump(encoders,f)
    data['cls_target'] = (data['efs']==0).astype('int')    
    

    
    sample_num = len(data['X'])
    cls_prediction = np.zeros(sample_num)
    categorical_cardinality = [len(data['X'][col].unique()) for col in data['feature_cat']]

    y_combine = data['efs'].astype('str')+'|'+data['race_group'].astype('str')
    skf = StratifiedKFold(n_splits=fold_n, shuffle=True, random_state=seed)
    for i,(train_index,eval_index) in enumerate(skf.split(data['X'],y_combine)):
        print('############fold',i)
        graph_data_train = create_graph(data,n_neighbors=args.n_neighbors_gnn,idx=train_index)
        graph_data_eval = create_graph(data,n_neighbors=args.n_neighbors_gnn,idx=eval_index)
        model = graphsage(continual_feature_num=len(data['feature_value']), 
                          hidden_channels=512, 
                          out_channels=1,
                          categorical_cardinality=categorical_cardinality,
                          embedding_dim=64,
                          embedding_projection_dim=128
                          ).to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.0005)
        
        # model train
        for epoch in range(70):
            model.train()
            optimizer.zero_grad()
            out = model(graph_data_train.x_cat,graph_data_train.x_cont, graph_data_train.edge_index, graph_data_train.edge_weight)
            loss = nn.BCELoss()(out,graph_data_train.y)
            loss.backward()
            optimizer.step()
            if epoch%1==0:
                model.eval()
                with torch.no_grad():
                    y_hat = model(graph_data_eval.x_cat,graph_data_eval.x_cont, graph_data_eval.edge_index,graph_data_eval.edge_weight)
                    y_hat = y_hat.cpu().numpy()
                    auc = roc_auc_score(graph_data_eval.y.cpu().numpy(), y_hat)
                print(f'epoch:{epoch}, auc:{auc}')
        model.eval()
        
        # fix the result shift between different fold
        rate_postive=sum(graph_data_train.y.cpu().numpy())/len(train_index)
        y_hat_raw = model(graph_data_train.x_cat,graph_data_train.x_cont, graph_data_train.edge_index,graph_data_train.edge_weight).cpu().detach().numpy()
        model.predict_shift = rate_postive/np.mean(y_hat_raw)

        # OOF predict  
        res_eval = model.predict_proba(graph_data_eval)[:,1]
        cls_prediction[eval_index] = res_eval

        # save model
        model_info = {'train_index':train_index,'eval_index':eval_index,'model':model}
        torch.save(model_info,args.model_dir+'gnn_cls_fold%s.pth'%(i))
    print('Over All OOF CLS-AUC:',roc_auc_score(data['cls_target'],cls_prediction))


if __name__=='__main__':
    gnn_cls_train(args.seed,args.fold_n)  