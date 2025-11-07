import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch import optim
import time
from torch.optim import lr_scheduler
import seaborn as sns
import pandas as pd
import argparse
import os
from dataloader import BinaryLoader
from loss import *
from tqdm import tqdm
import json
from model import ESPMedSAM
from functools import partial
import albumentations as A
from albumentations.pytorch.transforms import ToTensor
from torchmetrics.classification import BinaryAccuracy

# os.environ['CUDA_VISIBLE_DEVICES'] = '7'
torch.set_num_threads(4)

def fix_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()

def train_model(model, optimizer, scheduler, num_epochs=5):
    since = time.time()
    
    Loss_list = {'train': [], 'valid': []}
    Accuracy_list = {'train': [], 'valid': []}
    
    best_model_wts = model.state_dict()

    best_loss = float('inf')
    counter = 0
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train(True)
    
            else:
                model.train(False)  

            running_loss = []
            running_corrects_mask = []
            running_loss_patch = []
            running_corrects_patch = []
        
            # Iterate over data
            #for inputs,labels,label_for_ce,image_id in dataloaders[phase]: 
            for _, img, labels, img_id, mcls in tqdm(dataloaders[phase]):      
                # wrap them in Variable
                if torch.cuda.is_available():
    
                    img = Variable(img.cuda())
                    labels = Variable(labels.cuda())
                 
                else:
                    img, labels = Variable(img), Variable(labels)

                # label_patch = F.interpolate(labels, scale_factor=0.0625)
                label_patch = F.max_pool2d(labels, 32, 32)
                
                # zero the parameter gradients
                optimizer.zero_grad()

                for n, value in model.SemiTViT.named_parameters():
                    if f"med" in n:
                        value.requires_grad = True
                    else:
                        value.requires_grad = False

                for n, value in model.image_encoder.named_parameters():
                    value.requires_grad = False
                for n, value in model.prompt_encoder.named_parameters():
                    value.requires_grad = False
                for n, value in model.mask_decoder.named_parameters():
                    value.requires_grad = False
                for n, value in model.patch_decoder.named_parameters():
                    value.requires_grad = False
                for n, value in model.dense_prompter.named_parameters():
                    value.requires_grad = False
                for n, value in model.mask_query.named_parameters():
                    value.requires_grad = False

                model.apply(fix_bn)
                
                pred_mask, pred_patch, teacher_emb, student_emb = model(x=img, domain_seq=mcls)

                pred_mask = torch.sigmoid(pred_mask)
                pred_patch = torch.sigmoid(pred_patch)

                score_mask1 = iou_metric(pred_mask, labels)
                score_mask2 = acc_metric(pred_patch, label_patch)
                score_mask2 = torch.mean(score_mask2)



                loss = kd_loss(student_emb, teacher_emb)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                    
                # calculate loss and IoU
                running_loss.append(loss.item())
                running_corrects_mask.append(score_mask1.item())
                running_corrects_patch.append(score_mask2.item())
             

            epoch_loss = np.mean(running_loss)
            epoch_mask_iou = np.mean(running_corrects_mask)
            epoch_patch_acc = np.mean(running_corrects_patch)
            
            print('{} Loss: {:.4f} Mask IoU: {:.4f} Patch Acc: {:.4f}'.format(
                phase, np.mean(epoch_loss), np.mean(epoch_mask_iou), np.mean(epoch_patch_acc)))
            
            Loss_list[phase].append(epoch_loss)
            Accuracy_list[phase].append(epoch_mask_iou)

            # save parameters
            if phase == 'valid' and epoch_loss <= best_loss:
                best_loss = epoch_loss
                best_model_wts = model.state_dict()
                torch.save(best_model_wts, f'outputs/model_{args.dataset}_{epoch}.pth')
                counter = 0
            elif phase == 'valid' and epoch_loss > best_loss:
                counter += 1
            if phase == 'valid':
                scheduler.step()
        
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))
    


    return Loss_list, Accuracy_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str,default='ultrasound', help='dsb, isic, cvccdb, retinal, lung, ultrasound')
    parser.add_argument('--sam_pretrain', type=str,default='pretrain/mobile_sam.pt', help='/home/Qing_Xu/pretrain/mobile_sam.pt, efficient_sam_vitt, sam_vit_b_01ec64, outputs/model_decouple_domain_1.pth')
    parser.add_argument('--med_pretrain', type=str,default='outputs/model_best.pth', help='')
    parser.add_argument('--jsonfile', type=str,default='data_split.json', help='')
    parser.add_argument('--batch', type=int, default=4, help='batch size')
    parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
    parser.add_argument('--epoch', type=int, default=200, help='epoches')
    parser.add_argument('--size', type=float, default=1.0, help='epoches')
    args = parser.parse_args()

    os.makedirs(f'outputs/',exist_ok=True)

    args.jsonfile = f'datasets/{args.dataset}_data_split.json'
    
    with open(args.jsonfile, 'r') as f:
        df = json.load(f)

    val_files = df['valid']
    train_files = df['train']
    
    train_dataset = BinaryLoader(args.dataset, train_files, A.Compose([
        A.Resize(1024, 1024),
        A.HorizontalFlip(p=0.1),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensor()
        ], 
        additional_targets={'mask2': 'mask'}))
    val_dataset = BinaryLoader(args.dataset, val_files, A.Compose([
        A.Resize(1024, 1024),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensor()
        ],
        additional_targets={'mask2': 'mask'}))
    
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch, shuffle=True,drop_last=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=1)
    
    dataloaders = {'train':train_loader,'valid':val_loader}
    

    model = ESPMedSAM()

    pretrain_dict = torch.load(args.med_pretrain)
    selected_dict = {'.'.join(list(k.split('.'))[1:]): v for k, v in pretrain_dict.items() if list(k.split('.'))[0] == 'image_encoder'}
    model.image_encoder.load_state_dict(selected_dict, strict=True)

    selected_dict = {'.'.join(list(k.split('.'))[1:]): v for k, v in pretrain_dict.items() if list(k.split('.'))[0] == 'prompt_encoder'}
    model.prompt_encoder.load_state_dict(selected_dict, strict=True)

    selected_dict = {'.'.join(list(k.split('.'))[1:]): v for k, v in pretrain_dict.items() if list(k.split('.'))[0] == 'mask_decoder'}
    model.mask_decoder.load_state_dict(selected_dict, strict=True)

    selected_dict = {'.'.join(list(k.split('.'))[1:]): v for k, v in pretrain_dict.items() if list(k.split('.'))[0] == 'patch_decoder'}
    model.patch_decoder.load_state_dict(selected_dict, strict=True)

    selected_dict = {'.'.join(list(k.split('.'))[1:]): v for k, v in pretrain_dict.items() if list(k.split('.'))[0] == 'dense_prompter'}
    model.dense_prompter.load_state_dict(selected_dict, strict=True)

    selected_dict = {'.'.join(list(k.split('.'))[1:]): v for k, v in pretrain_dict.items() if list(k.split('.'))[0] == 'mask_query'}
    model.mask_query.load_state_dict(selected_dict, strict=True)

    pretrain_dict = torch.load(args.sam_pretrain)
    selected_dict = {'.'.join(list(k.split('.'))[1:]): v for k, v in pretrain_dict.items() if list(k.split('.'))[0] == 'image_encoder'}
    model.SemiTViT.load_state_dict(selected_dict, strict=False)



    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")

        model = nn.DataParallel(model)

    model = model.cuda()
        
    # Loss, IoU and Optimizer
    kd_loss = nn.MSELoss()
    # mask_loss = BinaryMaskLoss(0.8) # nn.CrossEntropyLoss()
    # patch_loss = nn.BCELoss() #BinaryBoundaryLoss()
    iou_metric = BinaryIoU()
    acc_metric = BinaryAccuracy(multidim_average='samplewise')

    optimizer = optim.Adam(model.parameters(),lr = args.lr)
    exp_lr_scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.98, verbose=True)
    Loss_list, Accuracy_list = train_model(model, optimizer, exp_lr_scheduler,
                           num_epochs=args.epoch)
    
    plt.title('Validation loss and IoU',)
    valid_data = pd.DataFrame({'Loss':Loss_list["valid"], 'IoU':Accuracy_list["valid"]})
    valid_data.to_csv(f'valid_data.csv')
    sns.lineplot(data=valid_data,dashes=False)
    plt.ylabel('Value')
    plt.xlabel('Epochs')
    plt.savefig('valid.png')
    
    plt.figure()
    plt.title('Training loss and IoU',)
    valid_data = pd.DataFrame({'Loss':Loss_list["train"],'IoU':Accuracy_list["train"]})
    valid_data.to_csv(f'train_data.csv')
    sns.lineplot(data=valid_data,dashes=False)
    plt.ylabel('Value')
    plt.xlabel('Epochs')
    plt.savefig('train.png')