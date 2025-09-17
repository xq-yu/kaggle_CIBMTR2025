import pickle
from sklearn.model_selection import StratifiedKFold
from torch import nn
import torch
from pytorch_lightning.utilities import grad_norm
import torch.nn as nn
from torch.utils.data import DataLoader,TensorDataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.metrics import roc_auc_score
from config import args
from tabm_reference import Model, make_parameter_groups
import rtdl_num_embeddings

def tabm_cls_seconde_process(data_file,label_encoders=None,imputers=None,scalers=None):
    with open(data_file,'rb') as f:
        data = pickle.load(f)
    input_columns = data['feature_cat']+data['feature_value']
    data['X'] = data['X'][input_columns]
    data['X']['is_cyto_score_same'] = (data['X']['cyto_score'] == data['X']['cyto_score_detail']).astype(int)
    data['feature_value']+=['is_cyto_score_same']
    data['X']['year_hct'] -= 2000

    if label_encoders is None:
        label_encoders = {col:LabelEncoder().fit(data['X'][col]) for col in data['feature_cat']}
    for col in data['feature_cat']:
        data['X'][col] = label_encoders[col].transform(data['X'][col])
    if imputers is None:
        imputers = {col:SimpleImputer(missing_values=np.nan, strategy='mean', add_indicator=False).fit(data['X'][[col]]) for col in data['feature_value']}
    for col in data['feature_value']:
        data['X'][col] = imputers[col].transform(data['X'][[col]])[:,0]
    if scalers is None:
        scalers = {col:StandardScaler().fit(data['X'][[col]]) for col in data['feature_value']}
    for col in data['feature_value']:
        data['X'][col] = scalers[col].transform(data['X'][[col]])[:,0]
    return data,label_encoders,imputers,scalers

class TabM_cls(pl.LightningModule):
    def __init__(
            self,
            config,
            bins,
            categorical_cardinality,
            continuous_dim,
            fea_cat,
            fea_cont
    ):
        super(TabM_cls, self).__init__()
        self.fea_cat=fea_cat
        self.fea_cont=fea_cont
        self.config = config
        self.model = Model(
                n_num_features=continuous_dim,
                cat_cardinalities=categorical_cardinality,
                n_classes=1,
                backbone=self.config['backbone'],
                bins=bins,
                num_embeddings=self.config['num_embeddings'],
                cat_embeddings={
                        'type': 'TrainablePositionEncoding',
                        'd_embedding' : [8] * len(categorical_cardinality),
                        'cardinality' : categorical_cardinality,
                },
                cat_dmodel = [8] * len(categorical_cardinality),
                arch_type=self.config['arch_type'],
                k=self.config['k'],
            )
        
        self.sigmoid = nn.Sigmoid()
        self.y = []
        self.y_hat=[]


    def on_before_optimizer_step(self, optimizer):
        """
        Compute the 2-norm for each layer
        If using mixed precision, the gradients are already unscaled here
        """
        norms = grad_norm(self.model, norm_type=2)
        self.log_dict(norms)

    def forward(self, x_cat, x_cont):
        """
        Forward pass that outputs the 1-dimensional prediction
        """
        x = self.model(x_cont,x_cat)
        x = x.squeeze(-1).mean(1)
        x = self.sigmoid(x)
        return x

    def training_step(self, batch, batch_idx):
        """
        defines how the model processes each batch of data during training.
        A batch is a combination of : categorical data, continuous data, efs_time (y) and efs event.
        y_hat is the efs_time prediction on all data and aux_pred is auxiliary prediction on embeddings.
        Calculates loss and race_group loss on full data.
        Auxiliary loss is calculated with an event mask, ignoring efs=0 predictions and taking the average.
        Returns loss and aux_loss multiplied by weight defined above.
        """
        x_cat, x_cont, y = batch
        y_hat = self(x_cat, x_cont)
        loss = nn.BCELoss()(y_hat,y)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        return loss
    

    def validation_step(self, batch, batch_idx):
        x_cat, x_cont, y = batch
        y_hat = self(x_cat, x_cont)
        loss = nn.BCELoss()(y_hat,y)
        self.log("validation_loss", loss)
        self.y_hat.append(y_hat.cpu().numpy())
        self.y.append(y.cpu().numpy())

    def on_validation_epoch_end(self):
        y_hat = np.concatenate(self.y_hat, axis=0)
        y = np.concatenate(self.y, axis=0)
        auc = roc_auc_score(y, y_hat)
        self.log("validation_auc", auc, on_epoch=True, prog_bar=True, logger=True)
        self.y_hat.clear()
        self.y.clear()


    def test_step(self, batch, batch_idx):
        x_cat, x_cont, y = batch
        y_hat = self(x_cat, x_cont)
        loss = nn.BCELoss()(y_hat,y)
        self.log("test_loss", loss)
        self.y_hat.append(y_hat.cpu().numpy())
        self.y.append(y.cpu().numpy())

    def on_test_epoch_end(self):
        y_hat = np.concatenate(self.y_hat, axis=0)
        y = np.concatenate(self.y, axis=0)
        auc = roc_auc_score(y, y_hat)
        self.log("test_auc", auc, on_epoch=True, prog_bar=True, logger=True)
        self.y_hat.clear()
        self.y.clear()
        
    def configure_optimizers(self):
        """
        Optimizer: AdamW optimizer with weight decay (L2 regularization).
        """
        optimizer = torch.optim.AdamW(make_parameter_groups(self), lr=self.config['lr'], weight_decay=self.config['weight_decay'])
        return {"optimizer": optimizer}
    
    def predict_proba(self,X):
        '''
        custom predict function
        input:
            X: pandas dataframe
        output:
            prob: np array of prob0 and prob1
        '''

        eval_set = TensorDataset(
            torch.tensor(np.array(X[self.fea_cat]), dtype=torch.long),
            torch.tensor(np.array(X[self.fea_cont]), dtype=torch.float32),
        )
        pred1= self(
            torch.tensor(eval_set.tensors[0], dtype=torch.long).to(args.device),
            torch.tensor(eval_set.tensors[1], dtype=torch.float32).to(args.device)
        ).cpu().detach().numpy()
        pred0 = 1-pred1
        prob = np.array([pred0,pred1]).T
        return prob

    
def train_tabm_cls(continuous_dim, train_set, eval_set, categorical_cardinality,fea_cat,fea_cont):
    """
    Defines model hyperparameters and fit the model.
    """
    
    
    # define model
    my_config = {
        'lr': 0.0005,
        'weight_decay': 0.000392,
        'k': 32,
        'num_embeddings': {
                    'type': 'PiecewiseLinearEmbeddings',
                    'd_embedding': 48,
                    'activation': True,
                    'version': 'A',
                },
        'backbone': {
                'type': 'MLP',
                'n_blocks': 2 ,
                'd_block': [512, 512],
                'dropout': 0.5425621416312014,
                'activation' : 'GELU',
            },
        'arch_type': 'tabm-mini',
        'n_bins' : 34
    } 
    bins = rtdl_num_embeddings.compute_bins(train_set.tensors[1])
    model = TabM_cls(config=my_config, 
                      bins=bins,
                      categorical_cardinality=categorical_cardinality,
                      continuous_dim=continuous_dim,
                      fea_cat=fea_cat,
                      fea_cont=fea_cont
                      )

    # define trainer
    trainer = pl.Trainer(
        accelerator='cuda',
        max_epochs=30,
        logger=False,
        check_val_every_n_epoch=1,
        enable_progress_bar=False
        
    )

    # define data
    dl_train = DataLoader(train_set, batch_size=2048, shuffle=True,drop_last=True)
    dl_val = DataLoader(eval_set, batch_size=2048, shuffle=True,drop_last=False)

    # train and validation
    trainer.fit(model, dl_train,dl_val)
    trainer.test(model, dl_val)
    return model.eval()

def tabm_cls_train(seed,fold_n):
    data,label_encoders,imputers,scalers = tabm_cls_seconde_process(args.data_process_dir+'data_train.pkl')
    encoders = {'label_encoders':label_encoders,'imputers':imputers,'scalers':scalers}
    with open(args.encoder_info_dir+'encoders_tabm_cls.pkl','wb') as f:
        pickle.dump(encoders,f)
    
    data['cls_target'] = (data['efs']==0).astype('int')

    sample_num = len(data['X'])
    cls_prediction = np.zeros(sample_num)

    y_combine = data['efs'].astype('str')+'|'+data['X']['race_group'].astype('str')
    skf = StratifiedKFold(n_splits=fold_n, shuffle=True, random_state=seed)
    for i,(train_index,eval_index) in enumerate(skf.split(data['X'],y_combine)):
        print('############fold',i)

        # create data
        X_train_cls = data['X'].iloc[train_index,:]
        y_train_cls = data['cls_target'][train_index]
        X_eval_cls = data['X'].iloc[eval_index,:]
        y_eval_cls = data['cls_target'][eval_index]
        train_set = TensorDataset(
            torch.tensor(np.array(X_train_cls[data['feature_cat']]), dtype=torch.long),
            torch.tensor(np.array(X_train_cls[data['feature_value']]), dtype=torch.float32),
            torch.tensor(np.array(y_train_cls), dtype=torch.float32),
        )
        eval_set = TensorDataset(
            torch.tensor(np.array(X_eval_cls[data['feature_cat']]), dtype=torch.long),
            torch.tensor(np.array(X_eval_cls[data['feature_value']]), dtype=torch.float32),
            torch.tensor(np.array(y_eval_cls), dtype=torch.float32),
        )
        categorical_cardinality = [len(data['X'][col].unique()) for col in data['feature_cat']]
        
        # model train
        model = train_tabm_cls(len(data['feature_value']), 
                               train_set, 
                               eval_set, 
                               categorical_cardinality,
                               fea_cat=data['feature_cat'],
                               fea_cont=data['feature_value'])
        
        #oof predict
        cls_prediction[eval_index] = model.to(args.device).predict_proba(data['X'].loc[eval_index])[:,1]

        model_info = {'train_index':train_index,'eval_index':eval_index,'model':model}
        torch.save(model_info,args.model_dir+'tabm_cls_fold%s.pth'%(i))
        with open(args.model_dir+'tabm_cls_fold%s.pkl'%(i),'wb') as f:
            pickle.dump(model_info,f)
        
    print('Over All OOF CLS-AUC:',roc_auc_score(data['cls_target'],cls_prediction))

if __name__=='__main__':
    tabm_cls_train(args.seed,args.fold_n)