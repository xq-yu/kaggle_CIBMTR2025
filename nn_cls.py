import pickle
from sklearn.model_selection import StratifiedKFold
from torch import nn
from typing import List
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,TensorDataset
from pytorch_tabular.models.common.layers import ODST
from pytorch_lightning.utilities import grad_norm
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import StochasticWeightAveraging
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.metrics import roc_auc_score
from config import args

def nn_cls_seconde_process(data_file,label_encoders=None,imputers=None,scalers=None):
    with open(data_file,'rb') as f:
        data = pickle.load(f)
    input_columns = data['feature_cat']+data['feature_value']
    data['X'] = data['X'][input_columns]
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


class CatEmbeddings_cls_nn(nn.Module):
    """
    Embedding module for the categorical dataframe.
    """
    def __init__(
        self,
        projection_dim: int,
        categorical_cardinality: List[int],
        embedding_dim: int
    ):
        """
        projection_dim: The dimension of the final output after projecting the concatenated embeddings into a lower-dimensional space.
        categorical_cardinality: A list where each element represents the number of unique categories (cardinality) in each categorical feature.
        embedding_dim: The size of the embedding space for each categorical feature.
        """
        super(CatEmbeddings_cls_nn, self).__init__()
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
    

class NN_cls(nn.Module):
    """
    Train a model on both categorical embeddings and numerical data.
    """
    def __init__(
            self,
            continuous_dim: int,
            categorical_cardinality: List[int],
            embedding_dim: int,
            projection_dim: int,
            hidden_dim: int,
            dropout: float = 0
    ):
        """
        continuous_dim: The number of continuous features.
        categorical_cardinality: A list of integers representing the number of unique categories in each categorical feature.
        embedding_dim: The dimensionality of the embedding space for each categorical feature.
        projection_dim: The size of the projected output space for the categorical embeddings.
        hidden_dim: The number of neurons in the hidden layer of the MLP.
        dropout: The dropout rate applied in the network.
        self.embeddings: previous embeddings for categorical data.
        self.mlp: defines an MLP model with an ODST layer followed by batch normalization and dropout.
        self.out: linear output layer that maps the output of the MLP to a single value
        self.dropout: defines dropout
        Weights initialization with xavier normal algorithm and biases with zeros.
        """
        super(NN_cls, self).__init__()
        self.embeddings = CatEmbeddings_cls_nn(projection_dim, categorical_cardinality, embedding_dim)
        self.mlp = nn.Sequential(
            ODST(projection_dim + continuous_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
        )
        self.out = nn.Linear(hidden_dim, 1)
        self.sig = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)

        # initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x_cat, x_cont):
        """
        Create embedding layers for categorical data, concatenate with continous variables.
        Add dropout and goes through MLP and return raw output and 1-dimensional output as well.
        """
        x = self.embeddings(x_cat)
        x = torch.cat([x, x_cont], dim=1)
        x = self.dropout(x)
        x = self.mlp(x)
        x = self.out(x)
        return self.sig(x)
    

class LitNN_cls(pl.LightningModule):
    """
    Main Model creation and losses definition to fully train the model.
    """
    def __init__(
            self,
            continuous_dim: int,
            categorical_cardinality: List[int],
            embedding_dim: int,
            projection_dim: int,
            hidden_dim: int,
            fea_cat:List[str],
            fea_value:List[str],
            lr: float = 5e-3,
            dropout: float = 0.2,
            weight_decay: float = 1e-3,
            aux_weight: float = 0.1,
            margin: float = 0.5,
            race_index: int = 0,

    ):
        """
        continuous_dim: The number of continuous input features.
        categorical_cardinality: A list of integers, where each element corresponds to the number of unique categories for each categorical feature.
        embedding_dim: The dimension of the embeddings for the categorical features.
        projection_dim: The dimension of the projected space after embedding concatenation.
        hidden_dim: The size of the hidden layers in the feedforward network (MLP).
        lr: The learning rate for the optimizer.
        dropout: Dropout probability to avoid overfitting.
        weight_decay: The L2 regularization term for the optimizer.
        aux_weight: Weight used for auxiliary tasks.
        margin: Margin used in some loss functions.
        race_index: An index that refer to race_group in the input data.
        """
        super(LitNN_cls, self).__init__()
        self.save_hyperparameters()

        # Creates an instance of the NN model defined above
        self.model = NN_cls(
            continuous_dim=self.hparams.continuous_dim,
            categorical_cardinality=self.hparams.categorical_cardinality,
            embedding_dim=self.hparams.embedding_dim,
            projection_dim=self.hparams.projection_dim,
            hidden_dim=self.hparams.hidden_dim,
            dropout=self.hparams.dropout
        )
        self.y = []
        self.y_hat=[]

        # Defines a small feedforward neural network that performs an auxiliary task with 1-dimensional output

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
        x = self.model(x_cat, x_cont)
        return x.squeeze(1)

    def training_step(self, batch, batch_idx):
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
        configures the optimizer and learning rate scheduler:
        * Optimizer: Adam optimizer with weight decay (L2 regularization).
        * Scheduler: Cosine Annealing scheduler, which adjusts the learning rate according to a cosine curve.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        
        scheduler_config = {
            "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=55,
                eta_min=6e-3
            ),
            "interval": "epoch",
            "frequency": 1,
            "strict": False,
        }
        

        return {"optimizer": optimizer, "lr_scheduler": scheduler_config}
    
    def predict_proba(self,X):
        eval_set = TensorDataset(
            torch.tensor(np.array(X[self.hparams.fea_cat]), dtype=torch.long),
            torch.tensor(np.array(X[self.hparams.fea_value]), dtype=torch.float32),
        )

        pred1= self(
            torch.tensor(eval_set.tensors[0], dtype=torch.long).to(args.device),
            torch.tensor(eval_set.tensors[1], dtype=torch.float32).to(args.device)
        ).cpu().detach().numpy()
        pred0 = 1-pred1
        return np.array([pred0,pred1]).T

def train_final_cls(continuous_dim, train_set, eval_set, categorical_cardinality,fea_cat,fea_value, hparams=None):
    """
    Defines model hyperparameters and fit the model.
    """
    if hparams is None:
        hparams = {
            "embedding_dim": 32,
            "projection_dim": 112,
            "hidden_dim": 56,
            "lr": 0.06464861983337984,
            "dropout": 0.05463240181423116,
            "aux_weight": 0.36545778308743806,
            "margin": 0.2588153271003354,
            "weight_decay": 0.0002773544957610778
        }
    model = LitNN_cls(
        continuous_dim=continuous_dim,
        categorical_cardinality=categorical_cardinality,
        fea_cat=fea_cat,
        fea_value=fea_value,
        **hparams
    )
    trainer = pl.Trainer(
        accelerator='cuda',
        max_epochs=55,
        logger=False,
        callbacks=[
            StochasticWeightAveraging(swa_lrs=1e-5, swa_epoch_start=40, annealing_epochs=15),
        ],
        check_val_every_n_epoch=1,
        enable_progress_bar=False
    )
    dl_train = DataLoader(train_set, batch_size=1024, shuffle=True,drop_last=True)
    dl_val = DataLoader(eval_set, batch_size=1024, shuffle=True,drop_last=False)

    trainer.fit(model, dl_train,dl_val)
    trainer.test(model, dl_val)
    return model.eval()


def nn_cls_train(seed,fold_n):
    data,label_encoders,imputers,scalers = nn_cls_seconde_process(args.data_process_dir+'data_train.pkl')
    encoders = {'label_encoders':label_encoders,'imputers':imputers,'scalers':scalers}
    with open(args.encoder_info_dir+'encoders_nn_cls.pkl','wb') as f:
        pickle.dump(encoders,f)
    data['cls_target'] = (data['efs']==0).astype('int')

    cls_prediction = np.zeros(len(data['X']))
    y_combine = data['efs'].astype('str')+'|'+data['X']['race_group'].astype('str')
    skf = StratifiedKFold(n_splits=fold_n, shuffle=True, random_state=seed)
    for i,(train_index,eval_index) in enumerate(skf.split(data['X'],y_combine)):
        print('############fold',i)
        # define data
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

        # train
        categorical_cardinality = [len(data['X'][col].unique()) for col in data['feature_cat']]
        model = train_final_cls(len(data['feature_value']), train_set, eval_set, categorical_cardinality,data['feature_cat'],data['feature_value'])
        model.eval()

        # oof predict
        cls_prediction[eval_index] = model.to(args.device).predict_proba(data['X'].loc[eval_index])[:,1]
    
        # save model
        model_info = {'train_index':train_index,'eval_index':eval_index,'model':model}
        torch.save(model_info,args.model_dir+'nn_cls_fold%s.pth'%(i))
        with open(args.model_dir+'nn_cls_fold%s.pkl'%(i),'wb') as f:
            pickle.dump(model_info,f)
        
    print('Over All OOF CLS-AUC:',roc_auc_score(data['cls_target'],cls_prediction))

if __name__=='__main__':
    nn_cls_train(args.seed,args.fold_n)