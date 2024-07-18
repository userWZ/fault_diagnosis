import os
import time

import torch
import numpy as np
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb

from dataloaders import get_dataloader
from models import construct_model
from utils.utils import local_directory
from utils import Logger
from utils import CalculateAccuracy, CalculatePrecision, CalculateRecall 
from utils import CalculateF1, CalculateCrossEntropy, CalculateMSE, CalculateMAE 


class Trainer:
    def __init__(self, model_cfg, dataset_cfg, logger_cfg,
                 train_cfg, name=None):
        # create local path and checkpoint directory
        self.local_path, self.ckpt_directory = local_directory(name, model_cfg, dataset_cfg, 'checkpoint')
        # setting seed
        self.seed = train_cfg.seed
        torch.manual_seed(self.seed)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() and train_cfg.use_gpu else 'cpu')
        self.model = construct_model(model_cfg)
        self.model.to(self.device)
        
        self.dataloader = get_dataloader(dataset_cfg)
        self.train_loader, self.test_loader = self.dataloader
        
        self.trainer_cfg = train_cfg
        self.logger = Logger(logger_cfg, local_path)
        self.epochs = train_cfg.epochs
        self.optimizer = torch.optim.Adam(self.model.parameters(), 
                                          lr=train_cfg.lr,
                                          betas=(train_cfg.momentum, train_cfg.beta),
                                          weight_decay=train_cfg.weight_decay)
        self.loss = train_cfg.loss
        # current iteration
        self.ckpt_epoch = train_cfg.ckpt_epoch
        
        # load checkpoint 
        if self.ckpt_epoch == 'max':
            self.ckpt_epoch = find_max_epoch(self.ckpt_directory)
        if self.ckpt_epoch >= 0:
            try:
                # load checkpoint file
                model_path = os.path.join(self.ckpt_directory, '{}.pkl'.format(self.ckpt_epochkpt_iter))
                checkpoint = torch.load(model_path, map_location='cpu')

                # feed model dict and optimizer state
                self.model.load_state_dict(checkpoint['model_state_dict'])
                if 'optimizer_state_dict' in checkpoint:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    # HACK to reset learning rate
                    self.optimizer.param_groups[0]['lr'] = learning_rate

                print('Successfully loaded model at iteration {}'.format(self.ckpt_epoch))
            except:
                print(f"Model checkpoint found at iteration {self.ckpt_epoch}, but was not successfully loaded - training from scratch.")
                self.ckpt_epoch = -1
        else:
            print('No valid checkpoint model found - training from scratch.')
            self.ckpt_epoch = -1
        
    def train(self) -> None: 
        cur_epoch= self.ckpt_epoch + 1
        while cur_epoch <= self.trainer_cfg.epochs:
            self.train_per_epoch(cur_epoch)
            if cur_epoch % self.trainer_cfg.val_freq == 0:
                self.val_per_epoch(cur_epoch)
            
            if cur_epoch % self.trainer_cfg.save_freq == 0:
                ckpt_name = '{}.pkl'.format(cur_epoch)
                torch.save({'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict()},
                           os.path.join(self.ckpt_directory, ckpt_name))         
            print('model at iteration %s is saved' % cur_epoch)
            self.logger.summary(cur_epoch)
            # self.logger.save_curves(cur_epoch)
            # self.logger.save_check_point(self.model, cur_epoch, step=5, state_dict=True)
            cur_epoch += 1
        
            
    def train_per_epoch(self, cur_epoch) ->None:
        self.model.train()
        epoch_loss = 0.0
        running_loss = 0.0
        for i, data in tqdm(enumerate(self.train_loader)):
            inputs, labels = data
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            outputs = self.model(inputs)
            
            # calculate metrics
            metrics = self.compute_metrics(outputs, labels, is_train=True)
            
            # back-propagation
            self.optimizer.zero_grad()
            loss = loss(metrics['train/' + self.loss])
            loss.backward()
            self.optimizer.step()
            
            epoch_loss += loss.item()
            running_loss += loss.item()
            
            # output to logger
            if i % self.trainer_cfg.logging_freq == 0:
                for key in metrics.keys():
                    self.logger.record_scalar(key, metrics[key])
            
            # print loss every print_freq
            if i % self.trainer_cfg.print_freq == 0:
                print(f'[{cur_epoch}, {i + 1}] loss: {running_loss / self.trainer_cfg.print_freq}')
                

    def val_per_epoch(self, epoch):
        self.model.eval()
        for i, data in enumerate(self.test_loader):
            inputs, labels = data
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            outputs = self.model(inputs)
            metrics = self.compute_metrics(outputs, labels, is_train=False)
            
            for key in metrics.keys():
                self.logger.record_scalar(key, metrics[key])
    
    def compute_metrics(self, pred, label, is_train=True):
        ce = CalculateCrossEntropy(label, pred)
        acc = CalculateAccuracy(label, pred)
        precision = CalculatePrecision(label, pred)
        recall = CalculateRecall(label, pred)
        f1 = CalculateF1(label, pred)
        mse = CalculateMSE(label, pred)
        mae = CalculateMAE(label, pred)
        
        prefix = 'train/' if is_train else 'val/'
        metrics = {
            prefix + 'cross_entropy': ce,
            prefix + 'accuracy': acc,
            prefix + 'precision': precision,
            prefix + 'recall': recall,
            prefix + 'f1': f1,
            prefix + 'mse': mse,
            prefix + 'mae': mae
        }
        
    def compute_loss(self, pred, label):
        pass
    
    def test(self):
        for i, data in enumerate(self.test_loader):
            inputs, labels = data
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            
            self.logger.record_scalar('test_loss', loss.item())
        

@hydra.main(version_base=None, config_path="configs/", config_name="config")        
def main(cfg: DictConfig):
    
    print(OmegaConf.to_yaml(cfg))
    OmegaConf.set_struct(cfg, False)  # Allow writing keys
    
    os.makedirs("exp/", exist_ok=True)
    
    trainer = Trainer(model_cfg=cfg.model, dataset_cfg=cfg.dataset, trainer_cfg=cfg.trainer)
    trainer.train()
    trainer.test()
    
    print("Training finished!")

if __name__ == '__main__':
    main()
        