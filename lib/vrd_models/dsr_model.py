import cv2
import numpy as np
import torch
import torch.nn as nn



import sys
import os.path as osp
from lib.network import FC, Conv2d, ROIPool, SemSim
from lib.network import batched_index_select, set_trainability
from easydict import EasyDict
import utils, globals

class DSRModel(nn.Module):
  def __init__(self, args):
    super(DSRModel, self).__init__()

    self.args = args

    # TODO: fix this cols thing, maybe rename it and also expand it
    self.x_cols = [self.args.feat_used.spat]

    if not hasattr(self.args, "n_obj"):
      raise ValueError("Can't build vrd model without knowing n_obj")
    if not hasattr(self.args, "n_pred"):
      raise ValueError("Can't build vrd model without knowing n_pred")


    # Size of the representation for each modality when fusing features
    self.args.n_fus_neurons = self.args.get("n_fus_neurons", 256)

    # Use batch normalization or not
    self.args.use_bn        = self.args.get("use_bn",        False)

    # Apply VGG net or directly receive the visual feature vector
    self.args.apply_vgg     = self.args.get("apply_vgg",     True) 

    if not self.args.feat_used.vis and self.args.feat_used.vis_so:
      raise ValueError("Can't use so features without visual features flag true") # TODO: fix

    self.total_fus_neurons = 0

    ###################
    # VISUAL FEATURES
    ###################
    if self.args.feat_used.vis:
      if self.args.apply_vgg:
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

      self.dropout0 = nn.Dropout()

      self.fc6    = FC(512 * 7 * 7, 4096)
      self.fc7    = FC(4096, 4096)
      self.fc_obj = FC(4096, self.args.n_obj, relu = False)
      set_trainability(self.fc_obj, requires_grad=False)

      # Load VGG layers: conv*, fc6, fc7
      self.load_pretrained_conv(osp.join(globals.data_dir, "VGG_imagenet.npy"), fix_layers=True)

      # Guide for jwyang's ROI Pooling Layers:
      #  https://medium.com/@andrewjong/how-to-use-roi-pool-and-roi-align-in-your-neural-networks-pytorch-1-0-b43e3d22d073
      self.roi_pool = ROIPool((7, 7), 1.0/16)

      self.fc8   = FC(4096, self.args.n_fus_neurons)
      self.total_fus_neurons += self.args.n_fus_neurons

      # using visual features of subject and object individually too
      if self.args.feat_used.vis_so:
        self.fc_so = FC(self.args.n_fus_neurons*2, self.args.n_fus_neurons)
        self.total_fus_neurons += self.args.n_fus_neurons

    ###################
    # SPATIAL FEATURES
    ###################
    if self.args.feat_used.spat == "dsr_spat_vec":
      self.fc_spatial  = FC(8, self.args.n_fus_neurons)
      self.total_fus_neurons += self.args.n_fus_neurons
    # using type 2 of spatial features
    elif self.args.feat_used.spat == "dsr_spat_mat": # TODO: test
      self.conv_spatial = nn.Sequential(Conv2d(2, 96, 5, same_padding=True, stride=2, bn=self.args.use_bn),
                                 Conv2d(96, 128, 5, same_padding=True, stride=2, bn=self.args.use_bn),
                                 Conv2d(128, 64, 8, same_padding=False, bn=self.args.use_bn))
      self.fc_spatial = FC(64, self.args.n_fus_neurons)
      self.total_fus_neurons += self.args.n_fus_neurons

    ###################
    # SEMANTIC FEATURES
    ###################
    if self.args.feat_used.sem:
      self.emb = nn.Embedding(self.args.n_obj, globals.emb_size)
      set_trainability(self.emb, requires_grad=False)
      self.fc_semantic = FC(globals.emb_size*2, self.args.n_fus_neurons)
      self.total_fus_neurons += self.args.n_fus_neurons

      # self.emb = nn.Embedding(self.n_obj, globals.emb_size)
      # set_trainability(self.emb, requires_grad=False)
      # self.fc_so_emb = FC(globals.emb_size*2, 256)

    if self.total_fus_neurons == 0:
      print("Warning! Using no features. The model is not expected to learn")
      self.total_fus_neurons = 1
    
    if self.args.n_pred >= 256:
      print("Warning! Perhaps this model has to be expanded for n_predicates > 256, to avoid 'information bottlenecks'")

    ######################################
    ######### PREDICATE SEMANTICS ########
    ######################################
    if not self.args.use_pred_sem:
      # Final layers
      self.fc_fusion = FC(self.total_fus_neurons, 256)
      self.fc_rel    = FC(256, self.args.n_pred, relu = False)
    else:
      assert self.args.pred_emb.shape[0] == self.args.n_pred
      mode = self.args.use_pred_sem-1
      # Two different ways of implementing the Semantic Similarity layer
      #  The input to the semantic similarity layer is a 300-dimensional semantic vector
      #  and the layer has to output one score per predicate (e.g 70 scores) given a similarity measure
      #  between two semantic vectors. The embeddings are 300-dimensional normalized vectors with values in [-1, 1]
      #  so the output of the previous layer is not activated with relu (which forces values to be non-negatives)
      if mode < 8:
        mode = mode
        # 2 Fully-Connected layers: 1024 -> 512 -> 300
        self.fc_fusion = FC(self.total_fus_neurons, 512)
        self.fc_rel    = nn.Sequential(
          FC(512, globals.emb_size, relu = False),
          SemSim(self.args.pred_emb, mode = mode),
        )
      elif mode < 16:
        mode = mode - 8
        # 1 Fully-Connected layers: 1024 -> 300
        self.fc_fusion = FC(self.total_fus_neurons, globals.emb_size, relu = False)
        self.fc_rel    = SemSim(self.args.pred_emb, mode = mode)
      else:
        mode = mode - 16
        from sklearn.metrics.pairwise import cosine_similarity

        pred2pred_sim = cosine_similarity(self.args.pred_emb, self.args.pred_emb)
        # pred2pred_sim = self.args.pP_prior
        # pred2pred_sim = cosine_similarity(self.args.pred_emb, self.args.pred_emb) + self.args.pP_prior

        print("Mode {} = {}{}{}{}{}".format(mode, mode%2, (mode//2)%2, (mode//4)%2, (mode//8)%2, (mode//16)%2))

        if mode % 2:
          pred2pred_sim = pred2pred_sim / np.linalg.norm(pred2pred_sim,axis=0)

        pred2pred_sim = torch.from_numpy(pred2pred_sim).to("cuda:0") # TODO fix

        # 1 Fully-Connected layers: 1024 -> 300
        if ((mode//16) % 2):
          self.fc_fusion = nn.Sequential(
            FC(self.total_fus_neurons, 256),
            FC(256, self.args.n_pred, relu = ((mode//2) % 2))
          )
        else:
          self.fc_fusion = FC(self.total_fus_neurons, self.args.n_pred, relu = ((mode//2)%2))

        self.fc_rel    = FC(self.args.n_pred, self.args.n_pred, relu = False, bias = False) # ((mode//4)%2))
        with torch.no_grad():
          self.fc_rel.fc.weight.data.copy_(pred2pred_sim)
        if ((mode//8)%2):
          self.fc_rel_not_trainable = True # TODO fix
          set_trainability(self.fc_rel, requires_grad = False)

  def forward(self, vis_features, obj_classes, obj_boxes, u_boxes, idx_s, idx_o, spat_features):

    n_batches = vis_features.size()[0]

    # turn our (batch_size×n×5) ROI into just (n×5)
    obj_boxes = obj_boxes.view(-1, obj_boxes.size()[2])
    u_boxes   =   u_boxes.view(-1,   u_boxes.size()[2])
    # reset ROI image-ID to align with the 0-indexed minibatch
    obj_boxes[:, 0] = obj_boxes[:, 0] - obj_boxes[0, 0]
    u_boxes[:, 0]   =   u_boxes[:, 0]  -  u_boxes[0, 0]

    n_objs = obj_boxes.size()[0]
    n_rels = u_boxes.size()[0]

    # Mmmm u_boxes.size()[1]
    x_fused = torch.empty((n_batches, n_rels, 0), device=utils.device)

    # print("u_boxes: ", u_boxes.shape)

    # VISUAL FEATURES
    if self.args.feat_used.vis:

      # ROI pooling for combined subjects' and objects' boxes

      # x_so = [self.roi_pool(x_img, obj_boxes[0]) for i in range(n_batches)]
      # x_so = [self.roi_pool(x_img, obj_boxes[0])]
      # x_so = torch.tensor(x_so).to(utils.device)


      # Visual features from the whole image
      # print("vis_features", vis_features.shape)
      if self.args.apply_vgg:
        x_img = self.conv1(vis_features)
        x_img = self.conv2(x_img)
        x_img = self.conv3(x_img)
        x_img = self.conv4(x_img)
        x_img = self.conv5(x_img)
      else:
        x_img = vis_features

      # print("x_img.shape: ", x_img.shape)
      # print("obj_boxes.shape: ", obj_boxes.shape)

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

      x_u = self.fc8(x_u)
      # print("x_fused: ", x_fused.shape)
      # print("x_u: ", x_u.shape)
      x_fused = torch.cat((x_fused, x_u), 2)
      # print("x_fused++: ", x_fused.shape)

      # using visual features of subject and object individually too
      if self.args.feat_used.vis_so:
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
    else:
      obj_scores = torch.zeros((n_objs, self.args.n_obj), device=utils.device)

    # SPATIAL FEATURES
    if self.args.feat_used.spat == "dsr_spat_vec":
      x_spat = self.fc_spatial(spat_features)
      x_fused = torch.cat((x_fused, x_spat), 2)
    elif self.args.feat_used.spat == "dsr_spat_mat":
      x_spat = self.conv_spatial(spat_features)
      x_spat = x_spat.view(n_batches, x_spat.size()[0], -1)
      # TODO: maybe this is actually the correct one: x_spat = x_spat.view(x_spat.size()[0], -1)
      x_spat = self.fc_spatial(x_spat)
      x_fused = torch.cat((x_fused, x_spat), 2)

    # SEMANTIC FEATURES
    if self.args.feat_used.sem:
      # x_sem  = self.fc_semantic(semantic_features)
      # x_fused = torch.cat((x_fused, x_sem), 2)

      # obj_classes in this case is simply a list of objects in the image
      # print()
      # print("obj_classes.shape: ", obj_classes.shape)
      emb = self.emb(obj_classes)
      # print("emb.shape: ", emb.shape)
      emb = torch.squeeze(emb, 0) # TODO: remove
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


    # FUSION
    if self.total_fus_neurons == 1:
      # print("Warning! Using no features. The model is not expected to learn")
      x_fused = torch.zeros((n_batches, n_rels, 1), device=utils.device)

    x_fused = self.fc_fusion(x_fused)
    rel_scores = self.fc_rel(x_fused)

    return obj_scores, rel_scores, x_fused


  def load_pretrained_conv(self, file_path, fix_layers=True):
    """ Load the weights for the initial convolutional layers and fully connecteda layers """

    vgg16_params = np.load(file_path, allow_pickle=True, encoding="latin1").item()
    # vgg16
    state_dict = self.state_dict()
    # print(state_dict)
    # print(params)

    if self.args.apply_vgg:
      for params_name, val in state_dict.items():
        if params_name.find("bn.") >= 0 or not "conv" in params_name or "spat" in params_name:
          continue
        # print(params_name, val)
        i, j = int(params_name[4]), int(params_name[6]) + 1
        # print(i, j)
        ptype = "weights" if params_name[-1] == "t" else "biases"
        key = "conv{}_{}".format(i, j)
        # print(ptype, key)
        param = torch.from_numpy(vgg16_params[key][ptype])
        # print(params[key][ptype])

        if ptype == "weights":
          param = param.permute(3, 2, 0, 1)

        val.copy_(param)

        #if fix_layers:
        #  set_trainability(utils.rgetattr(self, params_name), requires_grad=False)
      # TODO: fix this imprecision
      if fix_layers:
        set_trainability(self.conv1, requires_grad=False)
        set_trainability(self.conv2, requires_grad=False)
        set_trainability(self.conv3, requires_grad=False)
        set_trainability(self.conv4, requires_grad=False)
        set_trainability(self.conv5, requires_grad=False)


    # fc6 fc7
    pairs = {"fc6.fc": "fc6", "fc7.fc": "fc7"} # , "fc7_so.fc": "fc7"}
    for dest_layer, source_layer in pairs.items():
      # print(k,v)
      key = "{}.weight".format(dest_layer)
      # print(vgg16_params[v]["weights"])
      param = torch.from_numpy(vgg16_params[source_layer]["weights"]).permute(1, 0)
      state_dict[key].copy_(param)

      key = "{}.bias".format(dest_layer)
      # print(vgg16_params[v]["biases"])
      param = torch.from_numpy(vgg16_params[source_layer]["biases"])
      state_dict[key].copy_(param)

      if fix_layers:
        set_trainability(getattr(self, source_layer), requires_grad=False)


  def OriginalAdamOptimizer(self,
      lr = 0.00001,
      # momentum = 0.9,
      weight_decay = 0.0005,
      lr_fus_ratio = 10,
      lr_rel_ratio = 10
      ):

    # opt_params = list(self.parameters())
    opt_params = [
      {'params': self.fc_fusion.parameters(), 'lr': lr*lr_fus_ratio},
    ]
    if self.args.feat_used.vis:
      opt_params.append({'params': self.fc8.parameters(),          'lr': lr*10})
      if self.args.feat_used.vis_so:
        opt_params.append({'params': self.fc_so.parameters(),      'lr': lr*10})
    if self.args.feat_used.spat == "dsr_spat_vec":
      opt_params.append({'params': self.fc_spatial.parameters(),   'lr': lr*10})
    elif self.args.feat_used.spat == "dsr_spat_mat":
      opt_params.append({'params': self.conv_spatial.parameters(), 'lr': lr*10})
      opt_params.append({'params': self.fc_spatial.parameters(),   'lr': lr*10})
    if self.args.feat_used.sem:
      opt_params.append({'params': self.fc_semantic.parameters(),  'lr': lr*10})

    if not hasattr(self, "fc_rel_not_trainable") or not self.fc_rel_not_trainable:
      opt_params.append({'params': self.fc_rel.parameters(),       'lr': lr*lr_rel_ratio})

    return torch.optim.Adam(opt_params,
            lr = lr,
            weight_decay = weight_decay)
