import os
# pip install pytorch-lightning
# pip install pytorchvideo
import sys

import pandas as pd

import torch
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping

import warnings
warnings.filterwarnings('ignore')

import logging
logging.basicConfig(level=logging.DEBUG)

seed_everything(0)
torch.set_float32_matmul_precision('medium')

from model import VideoClassifier
from dataloader import train_dataloader, test_dataloader

BATCH_SIZE = 6 # 4

model = VideoClassifier(
    learning_rate = 1e-4,
    batch_size = BATCH_SIZE,
    num_worker = 0
)

checkpoint_callback = ModelCheckpoint(
    monitor = 'val_loss',
    dirpath = 'checkpoints',
    filename = 'model-{epoch:02d}-{val_loss:.2f}',
    save_top_k=1,
    mode='min',
    save_last=True,
)
lr_monitor = LearningRateMonitor(logging_interval='epoch')

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=3, # number of epochs with no improvement after which training will be stopped
    mode='min'
)

trainer = Trainer(
    max_epochs = 80,  # increased max_epochs to allow for early stopping
    accelerator = 'gpu',
    devices = -1, #-1
    precision = '16-mixed', # 16
    accumulate_grad_batches = 2,
    enable_progress_bar = True,
    num_sanity_val_steps = 0,
    callbacks = [lr_monitor, checkpoint_callback, early_stopping],
)

if __name__ == '__main__':
    train_loader = train_dataloader('train.csv', batch_size=BATCH_SIZE, num_workers=0)
    val_loader = test_dataloader('test.csv', batch_size=BATCH_SIZE, num_workers=0)
    test_loader = test_dataloader('valid.csv', batch_size=BATCH_SIZE, num_workers=0)
    
    trainer.fit(model, train_loader, val_loader) #ckpt_path='checkpoints/last.ckpt'
    trainer.validate(model, val_loader)
    
    trainer.test(model, test_loader)

    test_results = trainer.callback_metrics
    print(test_results)
    
    # Load the best checkpoint before saving the final model
    # best_model_path = checkpoint_callback.best_model_path
    # model.load_from_checkpoint(best_model_path)

    torch.save(model.state_dict(), "standard_final.pth")
    print("DONE!")