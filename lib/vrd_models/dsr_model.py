import cv2
import numpy as np
import torch
import torch.nn as nn



import sys
import os.path as osp
from lib.network import FC, Conv2d, ROIPool
from lib.network import batched_index_select, set_trainability
from easydict import EasyDict
import utils

class DSRModel(nn.Module):
  def __init__(self, args):
    super(DSRModel, self).__init__()

    self.args = args

    if not hasattr(self.args, "n_obj"):
      raise ValueError("Can't build vrd model without knowing n_obj")
    if not hasattr(self.args, "n_pred"):
      raise ValueError("Can't build vrd model without knowing n_pred")

    # This decides whether, in addition to the visual features of union box,
    #  those of subject and object individually are used or not
    self.args.use_so        = self.args.get("use_so",        True)

    # Use visual features
    self.args.use_vis       = self.args.get("use_vis",       True)

    # Use semantic features (TODO: this becomes the size of the semantic features)
    self.args.use_sem       = self.args.get("use_sem",       True)

    # Three types of spatial features:
    # - 0: no spatial info
    # - 1: 8-way relative location vector
    # - 2: dual mask
    self.args.use_spat      = self.args.get("use_spat",      1)

    self.args.use_pred_sem  = self.args.get("use_pred_sem",  False)

    # Size of the representation for each modality when fusing features
    self.args.n_fus_neurons = self.args.get("n_fus_neurons", 256)

    # Use batch normalization or not
    self.args.use_bn        = self.args.get("use_bn",        False)


    self.total_fus_neurons = 0



    self.conv1 = nn.Sequential(Conv2d(  3,  64, 3, same_padding=True, bn=self.args.use_bn),
                               Conv2d( 64,  64, 3, same_padding=True, bn=self.args.use_bn),
                               nn.MaxPool2d(2))
    self.conv2 = nn.Sequential(Conv2d( 64, 128, 3, same_padding=True, bn=self.args.use_bn),
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
    set_trainability(self.fc_obj, requires_grad=False)








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
      # self.fc_spatial = FC(64, self.args.n_fus_neurons)
      # self.total_fus_neurons += self.args.n_fus_neurons

    if(self.args.use_sem):
      # self.fc_semantic = FC(2*300, self.args.n_fus_neurons)
      # self.total_fus_neurons += self.args.n_fus_neurons
      self.emb = nn.Embedding(self.args.n_obj, 300)
      set_trainability(self.emb, requires_grad=False)
      self.fc_semantic = FC(300*2, 256)
      self.total_fus_neurons += self.args.n_fus_neurons

      # self.emb = nn.Embedding(self.n_obj, 300)
      # set_trainability(self.emb, requires_grad=False)
      # self.fc_so_emb = FC(300*2, 256)

    # Final layers
    self.fc_fusion = FC(self.total_fus_neurons, 256)

    output_size = self.args.n_pred
    if self.args.use_pred_sem:
      output_size = 300
    self.fc_rel    = FC(256, output_size, relu = False)

  def forward(self, img_blob, obj_boxes, u_boxes, idx_s, idx_o, spatial_features, obj_classes):






    n_batches = img_blob.size()[0]

    # Visual features from the whole image
    # print("img_blob", img_blob.shape)

    x_img = self.conv1(img_blob)
    x_img = self.conv2(x_img)
    x_img = self.conv3(x_img)
    x_img = self.conv4(x_img)
    x_img = self.conv5(x_img)

    # print("x_img.shape: ", x_img.shape)
    # print("obj_boxes.shape: ", obj_boxes.shape)

    # ROI pooling for combined subjects' and objects' boxes

    # x_so = [self.roi_pool(x_img, obj_boxes[0]) for i in range(n_batches)]
    # x_so = [self.roi_pool(x_img, obj_boxes[0])]
    # x_so = torch.tensor(x_so).to(utils.device)

    # turn our (batch_size×n×5) ROI into just (n×5)
    obj_boxes = obj_boxes.view(-1, obj_boxes.size()[2])
    u_boxes   =   u_boxes.view(-1,   u_boxes.size()[2])
    # reset ROI image-ID to align with the 0-indexed minibatch
    obj_boxes[:, 0] = obj_boxes[:, 0] - obj_boxes[0, 0]
    u_boxes[:, 0]   =   u_boxes[:, 0]  -  u_boxes[0, 0]
    # Warning: this is a critical point for batching with batchsize>1. Maybe this will cause trouble, together with the index_select down below
    x_so = self.roi_pool(x_img, obj_boxes) # .unsqueeze(0) ?

    # print("x_so.shape: ", x_so.shape)

    x_so = x_so.view(x_so.size()[0], -1)
    # print("x_so.shape: ", x_so.shape)

    x_so = self.fc6(x_so)
    x_so = self.dropout0(x_so)
    x_so = self.fc7(x_so)
    x_so = self.dropout0(x_so)
    obj_scores = self.fc_obj(x_so)

    # print("x_so.shape: ", x_so.shape)

    # ROI pooling for union boxes
    x_u = self.roi_pool(x_img, u_boxes).unsqueeze(0)
    # print("x_u.shape: ", x_u.shape)
    x_u = x_u.view(x_u.size()[0], x_u.size()[1], -1)
    # print("x_u.shape: ", x_u.shape)
    x_u = self.fc6(x_u)
    x_u = self.dropout0(x_u)
    x_u = self.fc7(x_u)
    x_u = self.dropout0(x_u)
    # print("x_u.shape: ", x_u.shape)

    # Mmmm u_boxes.size()[1]
    x_fused = torch.empty((n_batches, u_boxes.size()[0], 0)).to(utils.device)

    # print("u_boxes: ", u_boxes.shape)

    if(self.args.use_vis):
      x_u = self.fc8(x_u)
      # print("x_fused: ", x_fused.shape)
      # print("x_u: ", x_u.shape)
      x_fused = torch.cat((x_fused, x_u), 2)
      # print("x_fused++: ", x_fused.shape)

      # using visual features of subject and object individually too
      if(self.args.use_so):
        x_so = self.fc8(x_so)
        # print(x_so)
        # print(idx_s)
        # print()
        # print("x_so: ", x_so.shape)
        # print("idx_s: ", idx_s.shape)
        # print("idx_s: ", idx_s)

        x_s = torch.index_select(x_so, 0, idx_s[0]).unsqueeze(0) # TODO: warning, use batched_index_select otherwise this won't work
        x_o = torch.index_select(x_so, 0, idx_o[0]).unsqueeze(0) # TODO: warning, use batched_index_select otherwise this won't work
        # print()
        # print("x_s: ", x_s.shape)
        # print("x_o: ", x_o.shape)
        x_subobj = torch.cat((x_s, x_o), 2)
        # print("x_subobj: ", x_subobj.shape)
        x_subobj = self.fc_so(x_subobj)
        # print("x_subobj: ", x_subobj.shape)
        # print()
        x_fused = torch.cat((x_fused, x_subobj), 2)

    if(self.args.use_spat == 1):
      x_spat = self.fc_spatial(spatial_features)
      x_fused = torch.cat((x_fused, x_spat), 2)
    elif(self.args.use_spat == 2):
      raise NotImplementedError
      # lo = self.conv_lo(SpatialFea)
      # lo = lo.view(lo.size()[0], -1)
      # lo = self.fc_spatial(lo)
      # x_fused = torch.cat((x_fused, lo), 2)

    if(self.args.use_sem):
      # x_sem  = self.fc_semantic(semantic_features)
      # x_fused = torch.cat((x_fused, x_sem), 2)

      # obj_classes in this case is simply a list of objects in the image
      # print()
      # print("obj_classes.shape: ", obj_classes.shape)
      emb = self.emb(obj_classes)
      # print("emb.shape: ", emb.shape)
      emb = torch.squeeze(emb, 1)[0] # TODO: remove
      # print("emb.shape: ", emb.shape)
      # print("idx_s.shape: ", idx_s.shape)
      emb_subject = torch.index_select(emb, 0, idx_s[0]).unsqueeze(0) # TODO: warning, use batched_index_select otherwise this won't work
      emb_object  = torch.index_select(emb, 0, idx_o[0]).unsqueeze(0) # TODO: warning, use batched_index_select otherwise this won't work
      # print("emb_subject.shape: ", emb_subject.shape)
      emb_s_o = torch.cat((emb_subject, emb_object), dim=2)
      # print("emb_s_o.shape: ", emb_s_o.shape)
      emb = self.fc_semantic(emb_s_o)
      # print("emb.shape: ", emb.shape)
      x_fused = torch.cat((x_fused, emb), dim=2)
      # print("x_fused.shape: ", x_fused.shape)






    x_fused = self.fc_fusion(x_fused)
    rel_scores = self.fc_rel(x_fused)

    return obj_scores, rel_scores

  def load_pretrained_conv(self, file_path, fix_layers=True):
    """ Load the weights for the initial convolutional layers and fully connecteda layers """

    params = np.load(file_path, allow_pickle=True, encoding="latin1").item()
    # vgg16
    vgg16_dict = self.state_dict()
    # print(vgg16_dict)
    # print(params)

    # print()

    for params_name, val in vgg16_dict.items():
      if params_name.find("bn.") >= 0 or not "conv" in params_name or "spat" in params_name:
        continue
      # print(params_name, val)
      i, j = int(params_name[4]), int(params_name[6]) + 1
      # print(i, j)
      ptype = "weights" if params_name[-1] == "t" else "biases"
      key = "conv{}_{}".format(i, j)
      # print(ptype, key)
      param = torch.from_numpy(params[key][ptype])
      # print(params[key][ptype])

      if ptype == "weights":
        param = param.permute(3, 2, 0, 1)

      val.copy_(param)

      #if fix_layers:
      #  set_trainability(utils.rgetattr(self, params_name), requires_grad=False)
    # TODO: fix this imprecision
    set_trainability(self.conv1, requires_grad=False)
    set_trainability(self.conv2, requires_grad=False)
    set_trainability(self.conv3, requires_grad=False)
    set_trainability(self.conv4, requires_grad=False)
    set_trainability(self.conv5, requires_grad=False)


    # fc6 fc7
    frcnn_dict = self.state_dict()
    pairs = {"fc6.fc": "fc6", "fc7.fc": "fc7"}
    # pairs = {'fc6.fc': 'fc6', 'fc7.fc': 'fc7', 'fc7_so.fc': 'fc7'}
    for dest_layer, source_layer in pairs.items():
      # print(k,v)
      key = "{}.weight".format(dest_layer)
      # print(params[v]["weights"])
      param = torch.from_numpy(params[source_layer]["weights"]).permute(1, 0)
      frcnn_dict[key].copy_(param)

      key = "{}.bias".format(dest_layer)
      # print(params[v]["biases"])
      param = torch.from_numpy(params[source_layer]["biases"])
      frcnn_dict[key].copy_(param)

      if fix_layers:
        set_trainability(getattr(self, source_layer), requires_grad=False)

  def OriginalAdamOptimizer(self,
      lr = 0.00001,
      # momentum = 0.9,
      weight_decay = 0.0005,
      ):

    # opt_params = list(self.parameters())
    opt_params = [
      {'params': self.fc8.parameters(),       'lr': lr*10},
      {'params': self.fc_fusion.parameters(), 'lr': lr*10},
      {'params': self.fc_rel.parameters(),    'lr': lr*10},
    ]
    if(self.args.use_so):
      opt_params.append({'params': self.fc_so.parameters(), 'lr': lr*10})
    if(self.args.use_spat == 1):
      opt_params.append({'params': self.fc_spatial.parameters(), 'lr': lr*10})
    elif(self.args.use_spat == 2):
      raise NotImplementedError
      # opt_params.append({'params': self.conv_lo.parameters(), 'lr': lr*10})
      # opt_params.append({'params': self.fc_spatial.parameters(), 'lr': lr*10})
    if(self.args.use_sem):
      opt_params.append({'params': self.fc_semantic.parameters(), 'lr': lr*10})

    return torch.optim.Adam(opt_params,
            lr = lr,
            weight_decay = weight_decay)
