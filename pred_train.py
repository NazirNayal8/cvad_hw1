import torch
import torch.nn as nn
import wandb
import numpy as np
import matplotlib.pyplot as plt
import random
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from expert_dataset import ExpertDataset
from models.affordance_predictor import AffordancePredictor
from tqdm import tqdm
from datetime import datetime
from collections import namedtuple

use_wandb = True
global global_step, val_step
global_step = 0
val_step = 0

def validate(model, dataloader, reg_loss_func, cls_loss_func, hparams):
    """Validate model performance on the validation dataset"""
    # Your code here
    global val_step

    model.eval()
    loss_avg = 0
    count = 0

    with torch.no_grad():

        for img, c, ld, ra, tl_dist, tl_state in tqdm(dataloader, desc='Validation Loop'):

            img, c = img.to(hparams.device), c.to(hparams.device)
            ld, ra = ld.to(hparams.device), ra.to(hparams.device)
            tl_dist, tl_state = tl_dist.to(hparams.device), tl_state.to(hparams.device)

            ld_pred, ra_pred, tl_dist_pred, tl_state_pred = model(img, c)

            loss_ld = reg_loss_func(ld_pred.squeeze(), ld)
            loss_ra = reg_loss_func(ra_pred.squeeze(), ra)
            loss_tl_dist = reg_loss_func(tl_dist_pred.squeeze(), tl_dist)
            
            loss_tl_state = cls_loss_func(tl_state_pred, tl_state)

            loss = loss_ld + loss_ra + loss_tl_dist + loss_tl_state

            loss_avg += loss.item()
            count += 1

            if use_wandb:
                wandb.log({
                    'global_step': val_step,
                    'val/loss': loss.item(),
                    'val/lane_dist_loss_step': loss_ld.item(),
                    'val/route_angle_loss_step': loss_ra.item(),
                    'val/tl_dist_loss_step': loss_tl_dist.item(),
                    'val/tl_state_loss_step': loss_tl_state.item(),
                })

            val_step += 1
    
    loss_avg /= count

    return loss_avg

def train(model, dataloader, optimizer, reg_loss_func, cls_loss_func, hparams):
    """Train model on the training dataset for one epoch"""
    # Your code here
    global global_step

    model.train()
    loss_avg = 0
    count = 0

    for img, c, ld, ra, tl_dist, tl_state in tqdm(dataloader, desc='Training Loops'):

        img, c = img.to(hparams.device), c.to(hparams.device)
        ld, ra = ld.to(hparams.device), ra.to(hparams.device)
        tl_dist, tl_state = tl_dist.to(hparams.device), tl_state.to(hparams.device)

        optimizer.zero_grad()

        ld_pred, ra_pred, tl_dist_pred, tl_state_pred = model(img, c)

        loss_ld = reg_loss_func(ld_pred.squeeze(), ld)
        loss_ra = reg_loss_func(ra_pred.squeeze(), ra)
        loss_tl_dist = reg_loss_func(tl_dist_pred.squeeze(), tl_dist)
        
        loss_tl_state = cls_loss_func(tl_state_pred, tl_state)

        loss = loss_ld + loss_ra + loss_tl_dist + loss_tl_state

        loss.backward()
        optimizer.step()

        loss_avg += loss.item()
        count += 1

        if use_wandb:
            wandb.log({
                'global_step': global_step,
                'train/loss': loss.item(),
                'train/lane_dist_loss_step': loss_ld.item(),
                'train/route_angle_loss_step': loss_ra.item(),
                'train/tl_dist_loss_step': loss_tl_dist.item(),
                'train/tl_state_loss_step': loss_tl_state.item(),
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

    
    random.seed(datetime.now())
    GLOBAL_ID = random.randint(0, 1000000000)

    pl.seed_everything(2000)
    project_name = 'affordances_training'
    run_name = f'affordances_{GLOBAL_ID}'
    hparams = {
        'learning_rate': 2e-4,
        'weight_decay': 1e-4,
        'batch_size': 128,
        'num_epochs': 30,
        'save_path': 'model_logs/aff_cal_model/',
        'num_workers': 15,
        'dropout_p': 0.5,
        'img_size': (224, 224),
        'device': torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),
        'hidden_dim': 64,
    }

    if use_wandb:
        wandb.init(project=project_name, config=hparams, name=run_name)
        hparams = wandb.config
    else:
        hparams = namedtuple('hparams', hparams.keys())(*hparams.values())

    model = AffordancePredictor(hidden_dim=hparams.hidden_dim, dropout_p=hparams.dropout_p)
    model.to(hparams.device)

    reg_loss_func = nn.L1Loss()
    cls_loss_func = nn.CrossEntropyLoss()
    
    optimizer = Adam(model.parameters(), lr=hparams.learning_rate, weight_decay=hparams.weight_decay)

    train_dataset = ExpertDataset(train_root, img_size=hparams.img_size, mode='affordances')
    val_dataset = ExpertDataset(val_root, img_size=hparams.img_size, mode='affordances')

    # You can change these hyper parameters freely, and you can add more

    train_loader = DataLoader(train_dataset, batch_size=hparams.batch_size, shuffle=True,
                              drop_last=True, num_workers=hparams.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=hparams.batch_size, shuffle=False, num_workers=hparams.num_workers)

    train_losses = []
    val_losses = []
    for i in range(hparams.num_epochs):
        
        train_loss = train(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            reg_loss_func=reg_loss_func,
            cls_loss_func=cls_loss_func,
            hparams=hparams
        )
        train_losses.append(train_loss)

        val_loss = validate(
            model=model,
            dataloader=val_loader,
            reg_loss_func=reg_loss_func,
            cls_loss_func=cls_loss_func,
            hparams=hparams
        )

        val_losses.append(val_loss)

        if use_wandb:
            wandb.log({
                'epoch': i,
                'train/loss_epoch': train_loss,
                'val/loss_epoch': val_loss
            })
    
    
        torch.save(model, hparams.save_path + f'{run_name}_epoch_{i}.ckpt')
    #   plot_losses(train_losses, val_losses, save_name='testing_system')

if __name__ == "__main__":
    main()
