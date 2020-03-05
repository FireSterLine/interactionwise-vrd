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

        self.conv1 = nn.Sequential(Conv2d(3, 64, 3, same_padding=True, bn=bn),
                                   Conv2d(64, 64, 3, same_padding=True, bn=bn),
                                   nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(Conv2d(64, 128, 3, same_padding=True, bn=bn),
                                   Conv2d(128, 128, 3, same_padding=True, bn=bn),
                                   nn.MaxPool2d(2))

        self.conv3 = nn.Sequential(Conv2d(128, 256, 3, same_padding=True, bn=bn),
                                   Conv2d(256, 256, 3, same_padding=True, bn=bn),
                                   Conv2d(256, 256, 3, same_padding=True, bn=bn),
                                   nn.MaxPool2d(2))
        self.conv4 = nn.Sequential(Conv2d(256, 512, 3, same_padding=True, bn=bn),
                                   Conv2d(512, 512, 3, same_padding=True, bn=bn),
                                   Conv2d(512, 512, 3, same_padding=True, bn=bn),
                                   nn.MaxPool2d(2))
        self.conv5 = nn.Sequential(Conv2d(512, 512, 3, same_padding=True, bn=bn),
                                   Conv2d(512, 512, 3, same_padding=True, bn=bn),
                                   Conv2d(512, 512, 3, same_padding=True, bn=bn))


        self.fc_spatial  = FC(8, 256)
        self.fc_semantic = FC(2*300, 256)
        self.fc_fus1     = FC(256*2, 256)
        self.fc_fus2     = FC(256, self.n_pred, relu=False)


    def forward(self, img_blob, spatial_features, semantic_features):

        x_visual = self.conv1(im_blob)
        x_visual = self.conv2(x_visual)
        x_visual = self.conv3(x_visual)
        x_visual = self.conv4(x_visual)
        x_visual = self.conv5(x_visual)

        x_spat = self.fc_spatial(spatial_features)
        x_sem  = self.fc_semantic(semantic_features)

        x_fused = torch.cat((x_visual, x_spat, x_sem), 1)

        x_fused = self.fc_fus1(x_fused)
        x_fused = self.fc_fus2(x_fused)

        return x_fused

if __name__ == '__main__':
    m = vrd_model()
