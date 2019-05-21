# -*- coding: utf-8 -*-
"""
Created on Mon May 13 21:43:28 2019

@author: yj
"""

from config import *
from imdb import *

import pointSegNet

import os.path
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torch.backends.cudnn as cudnn

import numpy as np
try:
    from itertools import  ifilterfalse
except ImportError: # py3k
    from itertools import  filterfalse as ifilterfalse

from tensorboardX import SummaryWriter

#args={'csv_path':'../data/ImageSet/csv/', 'momentum':0.9, 'weight_decay':0.0001,
#      '}
args={'data_path':'../data', 'image_set':'train'}

writer=SummaryWriter()

class NormalLoss(nn.Module):
    def __init__(self, mc):
        super(NormalLoss, self).__init__()
        self.mc=mc
    def forward(self, outputs, labels, lidar_mask, loss_weight, lovasz=False):
        mc=self.mc
        if lovasz is False:
            targets=outputs.view(-1, mc.NUM_CLASS)
            labels=labels.view(-1)
            loss=F.cross_entropy(targets, labels)
        else:
            loss=lovasz_softmax(targets, labels)
        loss=loss*lidar_mask.view(-1,)
        loss=loss*loss_weight.view(-1,)
        loss=torch.sum(loss)/torch.sum(lidar_mask)
        return loss*mc.CLS_LOSS_COEF
 
    def lovasz_softmax(self, probas, labels, classes='present', per_image=False, ignore=None):
        """
        Multi-class Lovasz-Softmax loss
          probas: [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1).
                  Interpreted as binary (sigmoid) output with outputs of size [B, H, W].
          labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
          classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
          per_image: compute the loss per image instead of per batch
          ignore: void class labels
        """
        if per_image:
            loss = mean(lovasz_softmax_flat(*flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), ignore), classes=classes)
                              for prob, lab in zip(probas, labels))
        else:
            loss = lovasz_softmax_flat(*flatten_probas(probas, labels, ignore), classes=classes)
        return loss
    
    def isnan(self, x):
        return x != x
        
        
    def mean(self, l, ignore_nan=False, empty=0):
        """
        nanmean compatible with generators.
        """
        l = iter(l)
        if ignore_nan:
            l = ifilterfalse(isnan, l)
        try:
            n = 1
            acc = next(l)
        except StopIteration:
            if empty == 'raise':
                raise ValueError('Empty mean')
            return empty
        for n, v in enumerate(l, 2):
            acc += v
        if n == 1:
            return acc
        return acc / n
        
    def lovasz_grad(self, gt_sorted):
        """
        Computes gradient of the Lovasz extension w.r.t sorted errors
        See Alg. 1 in paper
        """
        p = len(gt_sorted)
        gts = gt_sorted.sum()
        intersection = gts - gt_sorted.float().cumsum(0)
        union = gts + (1 - gt_sorted).float().cumsum(0)
        jaccard = 1. - intersection / union
        if p > 1: # cover 1-pixel case
            jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
        return jaccard
    
    
    
    def iou(self, preds, labels, C, EMPTY=1., ignore=None, per_image=False):
        """
        Array of IoU for each (non ignored) class
        """
        if not per_image:
            preds, labels = (preds,), (labels,)
        ious = []
        for pred, label in zip(preds, labels):
            iou = []    
            for i in range(C):
                if i != ignore: # The ignored label is sometimes among predicted classes (ENet - CityScapes)
                    intersection = ((label == i) & (pred == i)).sum()
                    union = ((label == i) | ((pred == i) & (label != ignore))).sum()
                    if not union:
                        iou.append(EMPTY)
                    else:
                        iou.append(float(intersection) / float(union))
            ious.append(iou)
        ious = [mean(iou) for iou in zip(*ious)] # mean accross images if per_image
        return 100 * np.array(ious)
    
    
    
    
    
    def lovasz_softmax_flat(self, probas, labels, classes='present'):
        """
        Multi-class Lovasz-Softmax loss
          probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
          labels: [P] Tensor, ground truth labels (between 0 and C - 1)
          classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
        """
        if probas.numel() == 0:
            # only void pixels, the gradients should be 0
            return probas * 0.
        C = probas.size(1)
        losses = []
        class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
        for c in class_to_sum:
            fg = (labels == c).float() # foreground for class c
            if (classes is 'present' and fg.sum() == 0):
                continue
            if C == 1:
                if len(classes) > 1:
                    raise ValueError('Sigmoid output possible only with 1 class')
                class_pred = probas[:, 0]
            else:
                class_pred = probas[:, c]
            errors = (Variable(fg) - class_pred).abs()
            errors_sorted, perm = torch.sort(errors, 0, descending=True)
            perm = perm.data
            fg_sorted = fg[perm]
            losses.append(torch.dot(errors_sorted, Variable(lovasz_grad(fg_sorted))))
        return mean(losses)
    
    
    def flatten_probas(self, probas, labels, ignore=None):
        """
        Flattens predictions in the batch
        """
        if probas.dim() == 3:
            # assumes output of a sigmoid layer
            B, H, W = probas.size()
            probas = probas.view(B, 1, H, W)
        P, C = probas.size()
        # probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C = P, C
        labels = labels.view(-1)
        if ignore is None:
            return probas, labels
        valid = (labels != ignore)
        vprobas = probas[valid.nonzero().squeeze()]
        vlabels = labels[valid]
        return vprobas, vlabels
    
    def xloss(self, logits, labels, ignore=None):
        """
        Cross entropy loss
        """
        return F.cross_entropy(logits, Variable(labels), ignore_index=255)

