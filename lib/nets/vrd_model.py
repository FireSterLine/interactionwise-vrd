import cv2
import numpy as np
import torch
import torch.nn as nn

import sys
import os.path as osp

from lib.network import FC, Conv2d
from model.roi_layers.roi_pool import ROIPool

class vrd_model(nn.Module):
  def __init__(self, args):
    super(vrd_model, self).__init__()

    self.use_bn = False

    self.n_obj  = args.n_obj
    self.n_pred = args.n_pred


    self.conv1 = nn.Sequential(Conv2d(3, 64, 3, same_padding=True, bn=self.use_bn),
                               Conv2d(64, 64, 3, same_padding=True, bn=self.use_bn),
                               nn.MaxPool2d(2))
    self.conv2 = nn.Sequential(Conv2d(64, 128, 3, same_padding=True, bn=self.use_bn),
                               Conv2d(128, 128, 3, same_padding=True, bn=self.use_bn),
                               nn.MaxPool2d(2))

    self.conv3 = nn.Sequential(Conv2d(128, 256, 3, same_padding=True, bn=self.use_bn),
                               Conv2d(256, 256, 3, same_padding=True, bn=self.use_bn),
                               Conv2d(256, 256, 3, same_padding=True, bn=self.use_bn),
                               nn.MaxPool2d(2))
    self.conv4 = nn.Sequential(Conv2d(256, 512, 3, same_padding=True, bn=self.use_bn),
                               Conv2d(512, 512, 3, same_padding=True, bn=self.use_bn),
                               Conv2d(512, 512, 3, same_padding=True, bn=self.use_bn),
                               nn.MaxPool2d(2))
    self.conv5 = nn.Sequential(Conv2d(512, 512, 3, same_padding=True, bn=self.use_bn),
                               Conv2d(512, 512, 3, same_padding=True, bn=self.use_bn),
                               Conv2d(512, 512, 3, same_padding=True, bn=self.use_bn))

    # Guide for jwyang's ROI Pooling Layers:
    #  https://medium.com/@andrewjong/how-to-use-roi-pool-and-roi-align-in-your-neural-networks-pytorch-1-0-b43e3d22d073
    self.roi_pool = ROIPool((7, 7), 1.0/16)

    self.dropout0 = nn.Dropout()

    self.fc6    = FC(512 * 7 * 7, 4096)
    self.fc7    = FC(4096, 4096)
    self.fc_obj = FC(4096, self.n_obj, relu=False)

    self.fc_visual   = FC(4096, 256)

    self.fc_spatial  = FC(8, 256)
    self.fc_semantic = FC(2*300, 256)
    self.fc_fus1     = FC(256*2, 256)

    self.fc_rel     = FC(256, self.n_pred, relu=False)


  def forward(self, img_blob, so_boxes, spatial_features, semantic_features):

    # Visual features from the whole image

    x_img = self.conv1(img_blob)
    x_img = self.conv2(x_img)
    x_img = self.conv3(x_img)
    x_img = self.conv4(x_img)
    x_img = self.conv5(x_img)

    # ROI pooling combined for subjects' and objects' boxes
    # x_so = self.roi_pool(x_img, so_boxes)
    # x_so = x_so.view(x_so.size()[0], -1)
    # x_so = self.fc6(x_so)
    # x_so = self.dropout0(x_so)
    # x_so = self.fc7(x_so)
    # x_so = self.dropout0(x_so)
    # obj_scores = self.fc_obj(x_so)

    # ROI pooling for union boxes
    # x_u = self.roi_pool(x_img, u_boxes)
    # x_u = x_u.view(x_u.size()[0], -1)
    # x_u = self.fc6(x_u)
    # x_u = self.dropou0t(x_u)
    # x_u = self.fc7(x_u)
    # x_u = self.dropou0t(x_u)

    # x_so = self.fc_visual(x_so)
    # x_u = self.fc_visual(x_u)
    # x_vis = x_so + x_u...

    # x_vis = x_so

    # Fusion with spatial and semantic features

    x_spat = self.fc_spatial(spatial_features)
    x_sem  = self.fc_semantic(semantic_features)

    #print(x_vis.size())
    print(x_spat.size())
    print(x_sem.size())

    # Add x_vis ...
    x_fused = torch.cat((x_spat, x_sem), 1)

    x_fused = self.fc_fus1(x_fused)

    rel_scores = self.fc_rel(x_fused)

    # return obj_scores, rel_scores
    return rel_scores

if __name__ == '__main__':
  m = vrd_model()
