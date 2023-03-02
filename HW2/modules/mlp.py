from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics import Accuracy, ConfusionMatrix
import numpy as np

class MlpClassifier(pl.LightningModule):

    def __init__(self, hparams):
        super(MlpClassifier, self).__init__()
        self.save_hyperparameters(hparams)
        layers = [
            # TODO: define model layers here
            # Input self.hparams.num_features
            # Output self.hparams.num_classes
            nn.Linear(self.hparams.num_features, 256),
            torch.nn.ReLU(),
            nn.Linear(256, self.hparams.num_classes)
        ]
        self.model = nn.Sequential(*layers)
        self.loss = nn.CrossEntropyLoss()
        self.accuracy = Accuracy()
        self.temp = torch.zeros((15,15)).to(self.device)

    def forward(self, x):

        # for cnn
        # x = torch.squeeze(x, 2)
        # x = torch.squeeze(x, 2)
        
        # for cnn3d
        x = torch.squeeze(x, 3)
        x = torch.squeeze(x, 3)
        x = torch.squeeze(x, 3)
        x = torch.squeeze(x, 1)

        return self.model(x)


    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        acc = self.accuracy(y_hat, y)

        return loss

    def validation_step(self, batch, batch_idx):

        # self.temp = torch.zeros((15,15)).to(self.device)

        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        acc = self.accuracy(y_hat, y)   
        self.log_dict({'val_loss': loss, 'val_acc': acc},
                      on_epoch=True, prog_bar=True)

        confmat = ConfusionMatrix(num_classes=15).to(self.device)
        confmat = confmat(y_hat, y)

        temp_confmat = torch.tensor(confmat).to(self.device)
        self.temp = self.temp.to(self.device)
        self.temp = torch.add(self.temp, temp_confmat)
        self.temp = torch.tensor(self.temp, dtype=torch.int64)
        print(self.temp.tolist())
        
        

    def predict_step(self, batch, batch_idx):
        x, _ = batch
        y_hat = self(x)
        pred = y_hat.argmax(dim=-1)
        return pred

    def configure_optimizers(self):
        # TODO: define optimizer and optionally learning rate scheduler
        # The simplest form would be `return torch.optim.Adam(...)`
        # For more advanced usages, see https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        optimizer = torch.optim.Adam(self.model.parameters(),lr=self.hparams.learning_rate, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.7)
        return {"optimizer":optimizer, "lr_scheduler": scheduler}

    @classmethod
    def add_argparse_args(cls, parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--num_features', type=int)
        parser.add_argument('--num_classes', type=int, default=15)
        parser.add_argument('--learning_rate', type=float, default=0.001)
        parser.add_argument('--scheduler_factor', type=float, default=0.5)
        parser.add_argument('--scheduler_patience', type=int, default=10)
        return parser
