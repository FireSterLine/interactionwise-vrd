import cv2
import numpy as np
import torch
import torch.nn as nn

import sys
import os.path as osp

from lib.network import FC

class vrd_model(nn.Module):
    def __init__(self, args):
        super(vrd_model, self).__init__()

        self.n_obj  = args.n_obj
        self.n_pred = args.n_pred

        self.fc_spatial  = FC(8, 256)
        self.fc_semantic = FC(2*300, 256)
        self.fc_fus1     = FC(256*2, 256)
        self.fc_fus2     = FC(256, self.n_pred, relu=False)


    def forward(self, spatial_features, semantic_features):
        x_spat = self.fc_spatial(spatial_features)
        x_sem  = self.fc_semantic(semantic_features)

        x_fused = torch.cat((x_spat, x_sem), 1)

        x_fused = self.fc_fus1(x_fused)
        x_fused = self.fc_fus2(x_fused)

        return x_fused

if __name__ == '__main__':
    m = vrd_model()
