import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import os.path as osp

class FC(nn.Module):
    def __init__(self, in_features, out_features, relu=True):
        super(FC, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.fc(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class vrd_model(nn.Module):
    def __init__(self, args):
        super(vrd_model, self).__init__()

        self.n_obj  = args.n_obj
        self.n_pred = args.n_pred

        self.fc_spatial  = FC(8, 256)
        self.fc_semantic = FC(2*300, 256)
        self.fc_fus1     = FC(256*2, 256)
        self.fc_fus2     = FC(256, self.n_pred)


    def forward(self, spatial_features, sematic_features):

        x_spat = torch.Tensor(spatial_features) # Specify that it has to be on cuda (cuda() or is_cuda=True)
        x_sem  = torch.Tensor(sematic_features) # Specify that it has to be on cuda (cuda() or is_cuda=True)

        x_spat = self.fc_spatial(x_spat)
        x_sem  = self.fc_semantic(x_sem)

        x_fused = torch.cat((x_spat, x_sem), 1)

        x_fused = self.fc_fus1(x_fused)
        x_fused = self.fc_fus2(x_fused, relu=False)

        return x_fused

if __name__ == '__main__':
    m = vrd_model()
