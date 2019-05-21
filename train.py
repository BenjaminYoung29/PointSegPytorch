# -*- coding: utf-8 -*-
"""
Created on Mon May 13 22:14:45 2019

@author: yj
"""

import numpy as np



import torch

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim

from torchvision import datasets, transforms

import torch.backends.cudnn as cudnn



from imdb.dataset1 import *

from config.kitti_squeezeSeg_config import *
from pointSegNet import *
from squeezeSeg import *
#from pointSegNet1 import *
from util import *
import Loss



from tensorboardX import SummaryWriter

args={'csv_path':'/home/Job/ImageSet/csv/','data_path':'/home/Job/lidar_2d/','model_path':'./model',
     'lr':1e-3,'momentum':0.9,'weight_decay': 5e-4,'lr_step': 1000, 'lr_gamma':0.1, 'epochs':8,
     'start_epoch':0, 'pretrain':True, 'resume':True, 'gpu_ids':[0,1], 'batch_size':10}

writer=SummaryWriter()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(model, train_loader, criterion, optimizer, epoch):
    model.train()
    lovasz=False
    total_loss=0
    total_size =0
    total_tp = np.zeros(mc.NUM_CLASS)
    total_fp = np.zeros(mc.NUM_CLASS)
    total_fn = np.zeros(mc.NUM_CLASS)     
    print(len(train_loader))
    for batch_idx, datas in enumerate(train_loader):
        # trying to overfit a small data
        # if idx==100:
        #   break
        inputs, mask, labels, weight = datas
        inputs, mask, labels, weight = \
                inputs.to(device), mask.to(device), labels.to(device), weight.to(device)
        optimizer.zero_grad()              
        outputs=model(inputs)

        # _ is what?
        
        _, predicted=torch.max(outputs.data, 1)
        loss=criterion(outputs, labels, mask, weight, lovasz)

        writer.add_scalar('data/loss', loss/args['batch_size'], batch_idx*(epoch+1))
        loss.backward()
        optimizer.step()
        
        tp, fp, fn = evaluate(labels, predicted, mc.NUM_CLASS)
        
        total_tp += tp
        total_fp += fp
        total_fn += fn        
        
        total_loss+=loss.item()
        total_size += inputs.size(0)
        
        if batch_idx % 800 == 0:
            #now = datetime.datetime.now()

            print(f'[1] Train Epoch: {epoch} [{batch_idx * len(inputs)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tAverage loss: {total_loss / total_size:.6f}')
            #print(f'iou={Loss.NormalLoss.iou(predicted, labels)}')
            # TensoorboardX Save Input Image and Visualized Segmentation
            #writer.add_image('Input/Image/', (img_normalize(inputs[0, 3, :, :])).cpu(), batch_idx * (epoch+1))

            #writer.add_image('Predict/Image/', visualize_seg(predicted, mc)[0], batch_idx * (epoch+1))

            #writer.add_image('Target/Image/', visualize_seg(targets, mc)[0], batch_idx * (epoch+1))
            iou = total_tp / (total_tp+total_fn+total_fp+1e-12)
            precision = total_tp / (total_tp+total_fp+1e-12)
            recall = total_tp / (total_tp+total_fn+1e-12)

            print()
            print_evaluate(mc, 'IoU', iou)
            print_evaluate(mc, 'Precision', precision)
            print_evaluate(mc, 'Recall', recall)
            print()
            
            total_tp = np.zeros(mc.NUM_CLASS)
            total_fp = np.zeros(mc.NUM_CLASS)
            total_fn = np.zeros(mc.NUM_CLASS)
            
            if total_loss / total_size<=0.1:
                print(total_loss, total_size)
                lovasz=True
            

def test(mc, model, val_loader, epoch):

    model.eval()

    total_tp = np.zeros(mc.NUM_CLASS)
    total_fp = np.zeros(mc.NUM_CLASS)
    total_fn = np.zeros(mc.NUM_CLASS)

    with torch.no_grad():
        for batch_idx, datas in enumerate(val_loader):
            inputs, mask, targets, weight = datas
            inputs, mask, targets, weight = \
                    inputs.to(device), mask.to(device), targets.to(device), weight.to(device)

            outputs = model(inputs, mask)

            _, predicted = torch.max(outputs.data, 1)

            tp, fp, fn = evaluate(targets, predicted, mc.NUM_CLASS)
        
            total_tp += tp
            total_fp += fp
            total_fn += fn

        iou = total_tp / (total_tp+total_fn+total_fp+1e-12)
        precision = total_tp / (total_tp+total_fp+1e-12)
        recall = total_tp / (total_tp+total_fn+1e-12)

        print()
        print_evaluate(mc, 'IoU', iou)
        print_evaluate(mc, 'Precision', precision)
        print_evaluate(mc, 'Recall', recall)
        print()


if __name__=='__main__':
    torch.cuda.set_device(2)
    # torch.backends.cudnn.benchmark = True
    mc=kitti_squeezeSeg_config()
    
    if os.path.exists(args['model_path']) is False:
        os.mkdir(args['model_path'])
        
    train_datasets=KittiDataset(
            mc,
            csv_file=args['csv_path']+'train.csv',
            root_dir=args['data_path'],
            transform=transforms.Compose([transforms.ToTensor()])
            )
    train_dataloader=torch.utils.data.DataLoader(
            train_datasets,
            batch_size=args['batch_size'],
            shuffle=True,
            num_workers=0
            )
    print(len(train_dataloader))
    val_datasets=KittiDataset(
            mc,
            csv_file=args['csv_path']+'val.csv',
            root_dir=args['data_path'],
            transform=transforms.Compose([transforms.ToTensor()])
            )
    
    val_dataloader=torch.utils.data.DataLoader(
            val_datasets,
            batch_size=args['batch_size'],
            shuffle=True,
            num_workers=0
            )
    
    model=SqueeOri(mc).to(device)
    # model=SqueezeSeg(mc).to(device)
  
    criterion=Loss.NormalLoss(mc)
    
    optimizer=optim.SGD(model.parameters(), lr=args['lr'], momentum=args['momentum'], weight_decay=args['weight_decay'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args['lr_step'], gamma=args['lr_gamma'])

    model.cuda()
    
    for epoch in range(args['start_epoch'], args['epochs']):
        scheduler.step()
        print('-------------------------------------------------------------------')
        train(model, train_dataloader, criterion, optimizer, epoch)
        
        print('-------------------------------------------------------------------')
        print()


    writer.close()
