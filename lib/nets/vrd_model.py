import cv2
import numpy as np
import torch
import torch.nn as nn



import sys
import os.path as osp
from model.roi_layers.roi_pool import ROIPool
from lib.network import FC, Conv2d
from easydict import EasyDict


class vrd_model(nn.Module):
  def __init__(self, n_obj, n_pred,
      use_so = True,
      use_vis = True,
      use_spat = 1,
      use_sem = True,
      n_fus_neurons = 256,
      use_bn = False):
    super(vrd_model, self).__init__()

    self.args = EasyDict()

    self.args.n_obj  = n_obj
    self.args.n_pred = n_pred

    # This decides whether, in addition to the visual features of union box,
    #  those of subject and object individually are used or not
    self.args.use_so = use_so

    # Use semantic features (TODO: this becomes the size of the semantic features)
    self.args.use_sem = use_sem

    # Use visual features
    self.args.use_vis = use_vis

    # Three types of spatial features:
    # - 0: no spatial info
    # - 1: 8-way relative location vector
    # - 2: dual mask
    self.args.use_spat = use_spat

    # Size of the representation for each modality when fusing features
    self.args.n_fus_neurons = n_fus_neurons

    # Use batch normalization or not
    self.args.use_bn = use_bn



    self.total_fus_neurons = 0



    self.conv1 = nn.Sequential(Conv2d(3, 64, 3, same_padding=True, bn=self.args.use_bn),
                               Conv2d(64, 64, 3, same_padding=True, bn=self.args.use_bn),
                               nn.MaxPool2d(2))
    self.conv2 = nn.Sequential(Conv2d(64, 128, 3, same_padding=True, bn=self.args.use_bn),
                               Conv2d(128, 128, 3, same_padding=True, bn=self.args.use_bn),
                               nn.MaxPool2d(2))



    self.conv3 = nn.Sequential(Conv2d(128, 256, 3, same_padding=True, bn=self.args.use_bn),
                               Conv2d(256, 256, 3, same_padding=True, bn=self.args.use_bn),
                               Conv2d(256, 256, 3, same_padding=True, bn=self.args.use_bn),
                               nn.MaxPool2d(2))
    self.conv4 = nn.Sequential(Conv2d(256, 512, 3, same_padding=True, bn=self.args.use_bn),
                               Conv2d(512, 512, 3, same_padding=True, bn=self.args.use_bn),
                               Conv2d(512, 512, 3, same_padding=True, bn=self.args.use_bn),
                               nn.MaxPool2d(2))
    self.conv5 = nn.Sequential(Conv2d(512, 512, 3, same_padding=True, bn=self.args.use_bn),
                               Conv2d(512, 512, 3, same_padding=True, bn=self.args.use_bn),
                               Conv2d(512, 512, 3, same_padding=True, bn=self.args.use_bn))


    # Guide for jwyang's ROI Pooling Layers:
    #  https://medium.com/@andrewjong/how-to-use-roi-pool-and-roi-align-in-your-neural-networks-pytorch-1-0-b43e3d22d073
    self.roi_pool = ROIPool((7, 7), 1.0/16)

    self.dropout0 = nn.Dropout()

    self.fc6    = FC(512 * 7 * 7, 4096)
    self.fc7    = FC(4096, 4096)
    self.fc_obj = FC(4096, self.args.n_obj, relu = False)








    if(self.args.use_vis):
      self.fc8   = FC(4096, self.args.n_fus_neurons)
      self.total_fus_neurons += self.args.n_fus_neurons

      # using visual features of subject and object individually too
      if(self.args.use_so):
        self.fc_so = FC(self.args.n_fus_neurons*2, self.args.n_fus_neurons)
        self.total_fus_neurons += self.args.n_fus_neurons


    # using type 1 of spatial features
    if(self.args.use_spat == 1):
      self.fc_spatial  = FC(8, self.args.n_fus_neurons)
      self.total_fus_neurons += self.args.n_fus_neurons
    # using type 2 of spatial features
    elif(self.args.use_spat == 2):
      raise NotImplementedError()
      # self.conv_spat = nn.Sequential(Conv2d(2, 96, 5, same_padding=True, stride=2, bn=bn),
      #                            Conv2d(96, 128, 5, same_padding=True, stride=2, bn=bn),
      #                            Conv2d(128, 64, 8, same_padding=False, bn=bn))
      # self.fc_lov = FC(64, self.args.n_fus_neurons)
      # self.total_fus_neurons += self.args.n_fus_neurons

    if(self.args.use_sem):
      self.fc_semantic = FC(2*300, self.args.n_fus_neurons)
      # self.emb = nn.Embedding(self.n_obj, 300)
      # network.set_trainable(self.emb, requires_grad=False)
      # self.fc_so_emb = FC(300*2, 256)
      # self.total_fus_neurons += self.args.n_fus_neurons

    # Final layers
    self.fc_fusion = FC(self.total_fus_neurons, 256)
    self.fc_rel    = FC(256, self.args.n_pred, relu = False)

  def forward(self, img_blob, obj_boxes, u_boxes, idx_s, idx_o, spatial_features, semantic_features):








    # Visual features from the whole image

    x_img = self.conv1(img_blob)
    x_img = self.conv2(x_img)
    x_img = self.conv3(x_img)
    x_img = self.conv4(x_img)
    x_img = self.conv5(x_img)

    # ROI pooling for combined subjects' and objects' boxes
    x_so = self.roi_pool(x_img, obj_boxes)
    x_so = x_so.view(x_so.size()[0], -1)
    x_so = self.fc6(x_so)
    x_so = self.dropout0(x_so)
    x_so = self.fc7(x_so)
    x_so = self.dropout0(x_so)
    obj_scores = self.fc_obj(x_so)

    # ROI pooling for union boxes
    x_u = self.roi_pool(x_img, u_boxes)
    x_u = x_u.view(x_u.size()[0], -1)
    x_u = self.fc6(x_u)
    x_u = self.dropout0(x_u)
    x_u = self.fc7(x_u)
    x_u = self.dropout0(x_u)

    x_fused = torch.empty((u_boxes.size()[0], 0)).cuda()

    if(self.args.use_vis):
      x_u = self.fc8(x_u)
      x_fused = torch.cat((x_fused, x_u), 1)

      # using visual features of subject and object individually too
      if(self.args.use_so):
        x_so = self.fc8(x_so)
        x_s = torch.index_select(x_so, 0, idx_s)
        x_o = torch.index_select(x_so, 0, idx_o)
        x_so = torch.cat((x_s, x_o), 1)
        x_so = self.fc_so(x_so)
        x_fused = torch.cat((x_fused, x_so), 1)

    if(self.args.use_spat == 1):
      x_spat = self.fc_spatial(spatial_features)
      x = torch.cat((x_fused, x_spat), 1)
    elif(self.args.use_spat == 2):
      raise NotImplementedError
      # lo = self.conv_lo(SpatialFea)
      # lo = lo.view(lo.size()[0], -1)
      # lo = self.fc_lov(lo)
      # x = torch.cat((x_fused, lo), 1)

    # TODO: use embedding layer like they do
    if(self.args.use_sem):
      x_sem  = self.fc_semantic(semantic_features)
      x_fused = torch.cat((x_fused, x_sem), 1)






    x_fused = self.fc_fusion(x_fused)
    rel_scores = self.fc_rel(x_fused)

    return obj_scores, rel_scores

if __name__ == '__main__':
  m = vrd_model()
