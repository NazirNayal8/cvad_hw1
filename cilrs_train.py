from turtle import speed
from unicodedata import name
from regex import P
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import wandb
import random
from datetime import datetime
from torch.utils.data import DataLoader

from expert_dataset import ExpertDataset
from models.cilrs import CILRS

from torch.optim import Adam, SGD
from tqdm import tqdm
from collections import namedtuple

use_wandb = True
global global_step, val_step
global_step = 0
val_step = 0


def validate(model, dataloader, loss_func, hparams):
    """Validate model performance on the validation dataset"""
    global val_step
    # Your code here
    model.eval()
    loss_avg = 0
    count = 0
    with torch.no_grad():
        for img, v, c, a, in tqdm(dataloader, desc='Validation Loop'):

            img, v, c, a = img.to(hparams.device), v.to(hparams.device), c.to(hparams.device), a.to(hparams.device)

            throttle_pred, steer_pred, brake_pred, v_pred = model(img, v , c)

            loss_throttle = loss_func(throttle_pred.squeeze(), a[:, 0])
            loss_steer = loss_func(steer_pred.squeeze(), a[:, 1])
            loss_brake = loss_func(brake_pred.squeeze(), a[:, 2])
            loss_speed = loss_func(v_pred.squeeze(), v)

            loss = loss_throttle + hparams.steer_coeff * loss_steer + hparams.brake_coeff * loss_brake + hparams.speed_coeff * loss_speed

            loss_avg += loss.item()
            count += 1

            if use_wandb:
                wandb.log({
                    'global_step': val_step,
                    'val/loss': loss.item(),
                    'val/throttle_loss_step': loss_throttle.item(),
                    'val/steer_loss_step': loss_steer.item(),
                    'val/brake_loss_step': loss_brake.item(),
                    'val/speed_loss_step': loss_speed.item(),
                })

            val_step += 1

    loss_avg /= count

    return loss_avg

def train(model, dataloader, optimizer, loss_func, hparams):
    """Train model on the training dataset for one epoch"""
    # Your code here
    global global_step
    model.train()
    count = 0
    loss_avg = 0
    for img, v, c, a in tqdm(dataloader, desc='Training Loop'):

        img, v, c, a = img.to(hparams.device), v.to(hparams.device), c.to(hparams.device), a.to(hparams.device)

        optimizer.zero_grad()
        
        throttle_pred, steer_pred, brake_pred, v_pred = model(img, v , c)

        loss_throttle = loss_func(throttle_pred.squeeze(), a[:, 0])
        loss_steer = loss_func(steer_pred.squeeze(), a[:, 1])
        loss_brake = loss_func(brake_pred.squeeze(), a[:, 2])
        loss_speed = loss_func(v_pred.squeeze(), v)

        loss = loss_throttle + hparams.steer_coeff * loss_steer + hparams.brake_coeff * loss_brake + hparams.speed_coeff * loss_speed
    
        loss.backward()
        optimizer.step()

        count += 1
        loss_avg += loss.item()

        if use_wandb:

            wandb.log({
                'global_step': global_step,
                'train/loss_step': loss.item(),
                'train/throttle_loss_step': loss_throttle.item(),
                'train/steer_loss_step': loss_steer.item(),
                'train/brake_loss_step': loss_brake.item(),
                'train/speed_loss_step': loss_speed.item(),
            })

        global_step += 1

    loss_avg /= count
    return loss_avg



def plot_losses(train_loss, val_loss, save_name='run'):
    """Visualize your plots and save them for your report."""
    # Your code here
    
    plt.figure(figsize=(8, 5))
    plt.grid()
    plt.plot(np.arange(len(train_loss)), train_loss)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('')
    plt.show()
    plt.savefig(f'plots/{save_name}_train_loss.png')


    plt.figure(figsize=(8, 5))
    plt.grid()
    plt.plot(np.arange(len(val_loss)), val_loss)
    plt.title('Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('')
    plt.show()
    plt.savefig(f'plots/{save_name}_val_loss.png')




def main():
    # Change these paths to the correct paths in your downloaded expert dataset
    train_root = '/media/nazirnayal/DATA/datasets/Carla/expert_data/train'
    val_root = '/media/nazirnayal/DATA/datasets/Carla/expert_data/val'

    # You can change these hyper parameters freely, and you can add more
    random.seed(datetime.now())
    GLOBAL_ID = random.randint(0, 1000000000)

    pl.seed_everything(2000)
    project_name = 'cilrs_training'
    run_name = f'cilr_hyperopt_{GLOBAL_ID}_l1'
    hparams = {
        'learning_rate': 2e-4,
        'weight_decay': 1e-4,
        'batch_size': 128,
        'num_epochs': 30,
        'save_path': 'model_logs/cilrs_model',
        'speed_coeff': 0.5,
        'brake_coeff': 1.0,
        'steer_coeff': 1.0,
        'num_workers': 15,
        'dropout_p': 0.5,
        'img_size': (224, 224),
        'device': torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),
        'cond_module': 'branched'
    }
   

    if use_wandb:
        wandb.init(project=project_name, config=hparams, name=run_name)
        hparams = wandb.config
    else:
        hparams = namedtuple('hparams', hparams.keys())(*hparams.values())
    
    train_dataset = ExpertDataset(train_root, img_size=hparams.img_size)
    val_dataset = ExpertDataset(val_root, img_size=hparams.img_size)

    
    model = CILRS(cond_module=hparams.cond_module, dropout_p=hparams.dropout_p)
    model.to(hparams.device)

    loss_func = nn.L1Loss() # nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=hparams.learning_rate, weight_decay=hparams.weight_decay)

    train_loader = DataLoader(train_dataset, batch_size=hparams.batch_size, shuffle=True, num_workers=hparams.num_workers,
                              drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=hparams.batch_size, num_workers=hparams.num_workers, shuffle=False)

    
    train_losses = []
    val_losses = []
    for i in range(hparams.num_epochs):
        
        train_loss =  train(
            model=model, 
            dataloader=train_loader, 
            optimizer=optimizer, 
            loss_func=loss_func, 
            hparams=hparams
        )

        train_losses.append(train_loss)
        val_loss = validate(
            model=model, 
            dataloader=val_loader, 
            loss_func=loss_func,
            hparams=hparams
        )
        val_losses.append(val_loss)

        if use_wandb:
            wandb.log({
                'epoch': i,
                'train/loss_epoch': train_loss,
                'val/loss_epoch': val_loss
            })

        torch.save(model, hparams.save_path + f'_{run_name}_epoch_{i}.ckpt')
   #  plot_losses(train_losses, val_losses, save_name='testing_system')


if __name__ == "__main__":
    main()
