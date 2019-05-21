import pandas as pd

import numpy as np
import torch
from torch.utils.data import Dataset

import os
import sys

class KittiDataset(Dataset):
    def __init__(self, mc, csv_file, root_dir, transform=None):
        self.mc=mc
        self.lidar_2d_csv=pd.read_csv(csv_file)
        self.root_dir=root_dir
        self.transform=transform

    def __len__(self):
        return len(self.lidar_2d_csv)

    def __getitem__(self, idx):
        '''
        Returns:
        lidar: LiDAR input. Shape: batch x height x width x 5.
        lidar_mask: LiDAR mask, 0 for missing data and 1 otherwise.
            Shape: batch x height x width x 1.
        label_per_batch: point-wise labels. Shape: batch x height x width.
        weight_per_batch: loss weights for different classes. Shape: 
            batch x height x width
        '''
        mc=self.mc
        lidar_name=os.path.join(self.root_dir, self.lidar_2d_csv.iloc[idx, 0])
        record=np.load(lidar_name).astype(np.float32)

        if mc.DATA_AUGMENTATION:
            if mc.RANDOM_FLIPPING:
                if np.random.rand() > 0.5:
                    # flip y
                    record = record[:, ::-1, :]
                    record[:, :, 1] *= -1

        lidar=record[:,:,:5]

        lidar_mask=np.reshape(
            (lidar[:,:,4]>0)*1,
            [mc.ZENITH_LEVEL, mc.AZIMUTH_LEVEL, 1]
        )

        lidar=(lidar-mc.INPUT_MEAN)/mc.INPUT_STD

        label=record[:,:,5]
        
        weight=np.zeros(label.shape)
        for l in range(mc.NUM_CLASS):
            weight[label==l]=mc.CLS_LOSS_WEIGHT[int(l)]
            
        if self.transform:
            lidar=self.transform(lidar)
            lidar_mask=self.transform(lidar_mask)

        return (lidar.float(), lidar_mask.float(), torch.from_numpy(label.copy()).long, torch.from_numpy(weight.copy()).float())