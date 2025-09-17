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

def rankgnn_cls_seconde_process(data_file,label_encoders=None,imputers=None,scalers=None):
    with open(data_file,'rb') as f:
        data = pickle.load(f)
    input_columns = data['feature_cat']+data['feature_value']+data['feature_onehot']
    data['X'] = data['X'][input_columns]

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

class CatEmbeddings_cls_rankgnn(nn.Module):
    """
    Embedding module for the categorical dataframe.
    """
    def __init__(
        self,
        projection_dim: int,
        categorical_cardinality: List[int],
        embedding_dim: int
    ):
        super(CatEmbeddings_cls_rankgnn, self).__init__()
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

class Rankgraphsage(torch.nn.Module):
    def __init__(self, 
                 continual_feature_dim, 
                 hidden_channels, 
                 out_channels,
                 categorical_cardinality,
                 embedding_dim,
                 embedding_projection_dim,
                 ):
        super(Rankgraphsage, self).__init__()
        self.embeddings = CatEmbeddings_cls_rankgnn(embedding_projection_dim, categorical_cardinality, embedding_dim)
        self.conv1 = SAGEConv(continual_feature_dim+embedding_projection_dim, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, out_channels)

        #result shift/scale params
        self.predict_shift = None
        self.ran_min = None
        self.ran_max = None
    def forward(self, x_cat,x_cont, edge_index,edge_weight=None):
        x = self.embeddings(x_cat)
        x = torch.cat([x, x_cont], dim=1)
        x = F.gelu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.gelu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)
        return x[:,0]
    
    def predict_proba(self,graph_data):
        pred1= self(graph_data.x_cat,
            graph_data.x_cont,
            graph_data.edge_index,
            graph_data.edge_weight
        ).cpu().detach().numpy()
        if self.predict_shift is not None:
            pred1 = self.predict_shift*((pred1-self.ran_min)/(self.ran_max-self.ran_min))
        pred0 = 1-pred1
        return np.array([pred0,pred1]).T


def pairwise_logistic_loss(y_pred, y_true):
    """
    Pairwise Logistic Loss for binary classification.
    Args:
        y_pred (torch.Tensor): Predicted scores (logits) of shape (N,).
        y_true (torch.Tensor): True labels of shape (N,), where 1 is positive and 0 is negative.
    Returns:
        torch.Tensor: Ranking loss.
    """
    positive_indices = torch.where(y_true == 1)[0]
    negative_indices = torch.where(y_true == 0)[0]
    y_pos = y_pred[positive_indices].unsqueeze(1)  # Shape: (num_pos, 1)
    y_neg = y_pred[negative_indices].unsqueeze(0)  # Shape: (1, num_neg)
    loss = torch.log(1 + torch.exp(y_neg - y_pos)).mean()
    return loss


def rankgnn_cls_train(seed,fold_n):

    # data process
    data,label_encoders,imputers,scalers = rankgnn_cls_seconde_process(args.data_process_dir+'data_train.pkl')
    encoders = {'label_encoders':label_encoders,'imputers':imputers,'scalers':scalers}
    with open(args.encoder_info_dir+'encoders_rankgnn_cls.pkl','wb') as f:
        pickle.dump(encoders,f)
    categorical_cardinality = [len(data['X'][col].unique()) for col in data['feature_cat']]
    data['cls_target'] = (data['efs']==0).astype('int')    
    
    sample_num = len(data['X'])
    cls_prediction = np.zeros(sample_num)
    y_combine = data['efs'].astype('str')+'|'+data['X']['race_group'].astype('str')
    skf = StratifiedKFold(n_splits=fold_n, shuffle=True, random_state=seed) 
    for i,(train_index,eval_index) in enumerate(skf.split(data['X'],y_combine)):
        print('############fold',i)
        
        # define data
        graph_data_train = create_graph(data,n_neighbors=args.n_neighbors_rankgnn,idx=train_index)
        graph_data_eval = create_graph(data,n_neighbors=args.n_neighbors_rankgnn,idx=eval_index)
        
        # define model
        model = Rankgraphsage(continual_feature_dim=len(data['feature_value']), 
                          hidden_channels=256, 
                          out_channels=1,
                          categorical_cardinality=categorical_cardinality,
                          embedding_dim=64,
                          embedding_projection_dim=128,
                          ).to(args.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0002, weight_decay=0.005)

        # train
        for epoch in range(75):
            model.train()
            optimizer.zero_grad()
            out = model(graph_data_train.x_cat,graph_data_train.x_cont, graph_data_train.edge_index, graph_data_train.edge_weight)
            loss = pairwise_logistic_loss(out,graph_data_train.y)
            loss.backward()
            optimizer.step()
            
            # validation
            if epoch%1==0:
                model.eval()
                with torch.no_grad():
                    y_hat = model(graph_data_eval.x_cat,graph_data_eval.x_cont, graph_data_eval.edge_index,graph_data_eval.edge_weight)
                    y_hat = y_hat.cpu().numpy()
                    auc = roc_auc_score(graph_data_eval.y.cpu().numpy(), y_hat)
                print(f'epoch:{epoch}, auc:{auc}')
        model.eval()
        
        # scale to 0~1 and fix the result shift between different fold
        y_train_cls = graph_data_train.y.cpu().numpy()
        rate_postive=sum(y_train_cls)/len(train_index) 
        y_hat_raw = model(graph_data_train.x_cat,graph_data_train.x_cont, graph_data_train.edge_index,graph_data_train.edge_weight).cpu().detach().numpy()
        model.ran_min = np.min(y_hat_raw)
        model.ran_max = np.max(y_hat_raw)
        y_hat_scaled = (y_hat_raw-model.ran_min)/(model.ran_max-model.ran_min)
        model.predict_shift = rate_postive/np.mean(y_hat_scaled)

        # OOF predict
        res_eval = model.predict_proba(graph_data_eval)[:,1]
        cls_prediction[eval_index] = res_eval

        # save model
        model_info = {'train_index':train_index,'eval_index':eval_index,'model':model}
        torch.save(model_info,args.model_dir+'rankgnn_cls_fold%s.pth'%(i))
        with open(args.model_dir+'rankgnn_cls_fold%s.pkl'%(i),'wb') as f:
            pickle.dump(model_info,f)

    print('Over All OOF CLS-AUC:',roc_auc_score(data['cls_target'],cls_prediction))

if __name__=='__main__':
    rankgnn_cls_train(args.seed,args.fold_n)