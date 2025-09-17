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

class RankTabM_cls(pl.LightningModule):
    def __init__(
            self,
            config,
            bins,
            categorical_cardinality,
            continuous_dim,
            fea_cat,
            fea_cont
    ):
        super(RankTabM_cls, self).__init__()
        self.fea_cat = fea_cat
        self.fea_cont = fea_cont
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
        self.y = []
        self.y_hat=[]

        self.predict_shift=None
        self.ran_min = None
        self.ran_max = None

    def on_before_optimizer_step(self, optimizer):
        """
        Compute the 2-norm for each layer
        If using mixed precision, the gradients are already unscaled here
        """
        norms = grad_norm(self.model, norm_type=2)
        self.log_dict(norms)

    def forward(self, x_cat, x_cont):
        """
        Forward pass that outputs the 1-dimensional prediction and the embeddings (raw output)
        """
        x = self.model(x_cont,x_cat)
        x = x.squeeze(-1).mean(1)
        return x

    def training_step(self, batch, batch_idx):
        """
        Defines how the model processes each batch of data during training.
        A batch is a combination of: categorical data, continuous data, and target labels.
        Calculates the pairwise logistic loss.
        """
        x_cat, x_cont, y = batch
        y_hat = self(x_cat, x_cont)
        loss = self.pairwise_logistic_loss(y_hat,y)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def pairwise_logistic_loss(self,y_pred, y_true):
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
        loss = torch.relu(y_neg - y_pos+0.5).mean()
        return loss

    def validation_step(self, batch, batch_idx):
        x_cat, x_cont, y = batch
        y_hat = self(x_cat, x_cont)
        loss = self.pairwise_logistic_loss(y_hat,y)
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
        loss = self.pairwise_logistic_loss(y_hat,y)
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
        * Optimizer: AdamW optimizer with weight decay (L2 regularization).
        """
        optimizer = torch.optim.AdamW(make_parameter_groups(self), lr=self.config['lr'], weight_decay=self.config['weight_decay'])
        return {"optimizer": optimizer}
    
    def predict_proba(self,X):
        eval_set = TensorDataset(
            torch.tensor(np.array(X[self.fea_cat]), dtype=torch.long),
            torch.tensor(np.array(X[self.fea_cont]), dtype=torch.float32),
        )

        pred1= self(
            torch.tensor(eval_set.tensors[0], dtype=torch.long).to(args.device),
            torch.tensor(eval_set.tensors[1], dtype=torch.float32).to(args.device)
        ).cpu().detach().numpy()


        if self.predict_shift is not None:
            pred1 = self.predict_shift*((pred1-self.ran_min)/(self.ran_max-self.ran_min))
        pred0 = 1-pred1
        return np.array([pred0,pred1]).T


def train_ranktabm_cls(continuous_dim, train_set, eval_set, categorical_cardinality,fea_cat,fea_cont):
    """
    Defines model hyperparameters and fit the model.
    """
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
    model = RankTabM_cls(config=my_config, 
                      bins=bins,
                      categorical_cardinality=categorical_cardinality,
                      continuous_dim=continuous_dim,
                      fea_cat = fea_cat,
                      fea_cont = fea_cont
                      )

    trainer = pl.Trainer(
        accelerator='cuda',
        max_epochs=40,
        logger=False,
        callbacks=[
        ],
        check_val_every_n_epoch=1,
        enable_progress_bar=False
        
    )
    dl_train = DataLoader(train_set, batch_size=2048, shuffle=True,drop_last=True)
    dl_val = DataLoader(eval_set, batch_size=2048, shuffle=True,drop_last=False)

    trainer.fit(model, dl_train,dl_val)
    trainer.test(model, dl_val)
    trainer.test(model, dl_train)
    return model.eval()


def ranktabm_cls_seconde_process(data_file,label_encoders=None,imputers=None,scalers=None):
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



def ranktabm_cls_train(seed,fold_n):
    data,label_encoders,imputers,scalers = ranktabm_cls_seconde_process(args.data_process_dir+'data_train.pkl')
    encoders = {'label_encoders':label_encoders,'imputers':imputers,'scalers':scalers}
    with open(args.encoder_info_dir+'encoders_ranktabm_cls.pkl','wb') as f:
        pickle.dump(encoders,f)
    data['cls_target'] = (data['efs']==0).astype('int')

    sample_num = len(data['X'])
    cls_prediction = np.zeros(sample_num)

    y_combine = data['efs'].astype('str')+'|'+data['X']['race_group'].astype('str')
    skf = StratifiedKFold(n_splits=fold_n, shuffle=True, random_state=seed)
    for i,(train_index,eval_index) in enumerate(skf.split(data['X'],y_combine)):
        print('############fold',i)
        
        # data process
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
        
        # train
        model = train_ranktabm_cls(len(data['feature_value']), train_set, eval_set, categorical_cardinality,data['feature_cat'],data['feature_value'])
        model.eval()
        
        # fix the result shift between different fold
        rate_postive=sum(y_train_cls)/len(train_index)  
        y_hat_raw = np.zeros(len(train_index))
        train_set = TensorDataset(
            torch.tensor(np.array(X_train_cls[data['feature_cat']]), dtype=torch.long),
            torch.tensor(np.array(X_train_cls[data['feature_value']]), dtype=torch.float32),
            torch.tensor(np.array(y_train_cls), dtype=torch.float32),
            torch.tensor(np.array(range(len(y_train_cls))),dtype=torch.long)
        )
        dl_train = DataLoader(train_set, batch_size=2048, shuffle=False,drop_last=False)
        for batch in dl_train:
            idx = batch[3].cpu().numpy()
            y_hat_raw[idx] = model(batch[0],batch[1]).cpu().detach().numpy()
        model.ran_min = np.min(y_hat_raw)
        model.ran_max = np.max(y_hat_raw)
        y_hat_scaled = (y_hat_raw-model.ran_min)/(model.ran_max-model.ran_min)
        model.predict_shift = rate_postive/np.mean(y_hat_scaled)

        # OOF predict
        cls_prediction[eval_index] = model.to(args.device).predict_proba(data['X'].loc[eval_index])[:,1]
    
        # save model
        model_info = {'train_index':train_index,'eval_index':eval_index,'model':model}
        torch.save(model_info,args.model_dir+'ranktabm_cls_fold%s.pth'%(i))
        with open(args.model_dir+'ranktabm_cls_fold%s.pkl'%(i),'wb') as f:
            pickle.dump(model_info,f)

    print('Over All OOF CLS-AUC:',roc_auc_score(data['cls_target'],cls_prediction))

if __name__=='__main__':
    ranktabm_cls_train(args.seed,args.fold_n)