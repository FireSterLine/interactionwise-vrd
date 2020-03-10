import os
import os.path as osp
import sys
import pickle
import argparse
from tabulate import tabulate
import time

import torch
import torch.nn as nn
import torch.nn.init

from easydict import EasyDict
import globals, utils
from lib.nets.vrd_model import vrd_model
from lib.datalayers import VRDDataLayer
from model.utils.net_utils import weights_normal_init, save_checkpoint
from lib.evaluation_dsr import eval_recall_at_N, eval_obj_img # TODO remove this module
#, save_net, load_net, \
#      adjust_learning_rate, , clip_gradient

# from lib.model import train_net, test_pre_net, test_rel_net



































class vrd_trainer():

  def __init__(self, dataset_name="vrd", pretrained=False): # pretrained="epoch_4_checkpoint.pth.tar"):

    print("vrd_trainer() called with args:")
    print([dataset_name, pretrained])

    self.session_name = "test" # Training session name?
    self.start_epoch = 0
    self.max_epochs = 10 # 200
    self.checkpoint_frequency = 10

    # Does this have to be a constant?
    self.iters_per_epoch = 10

    self.batch_size = 1 # TODO
    self.num_workers = 0

    self.save_dir = osp.join(globals.models_dir, self.session_name)
    if not osp.exists(self.save_dir):
      os.mkdir(self.save_dir)

    self.dataset_name = dataset_name
    self.dataset_args = {"ds_name" : self.dataset_name, "with_bg_obj" : False, "with_bg_pred" : False}
    self.pretrained = pretrained

    # load data layer
    print("Initializing data layer...")
    # self.datalayer = VRDDataLayer({"ds_name" : self.dataset_name, "with_bg_obj" : True}, "train")
    self.datalayer = VRDDataLayer(self.dataset_args, "train")
    # self.datalayer = VRDDataLayer(self.dataset_name, "train")

    # TODO: Pytorch DataLoader()
    # self.dataset = VRDDataset()
    # self.datalayer = torch.utils.data.DataLoader(self.dataset,
    #                  batch_size=self.batch_size,
    #                  # sampler= Random ...,
    #                  num_workers=self.num_workers)

    load_pretrained = isinstance(self.pretrained, str)

    print("Initializing VRD Model...")

    self.net_args = EasyDict()

    self.net_args.n_obj  = self.datalayer.n_obj
    self.net_args.n_pred = self.datalayer.n_pred

    # This decides whether, in addition to the visual features of union box,
    #  those of subject and object individually are used or not
    self.net_args.use_so = True

    # Use visual features
    self.net_args.use_vis = True

    # Use semantic features (TODO: this becomes the size of the semantic features)
    self.net_args.use_sem = True

    # Three types of spatial features:
    # - 0: no spatial info
    # - 1: 8-way relative location vector
    # - 2: dual mask
    self.net_args.use_spat = 0

    # Size of the representation for each modality when fusing features
    self.net_args.n_fus_neurons = 256

    # Use batch normalization or not
    self.net_args.use_bn = False

    # initialize the model using the args set above
    print("Args: ", self.net_args)
    self.net = vrd_model(self.net_args) # TODO: load_pretrained affects how the model is initialized?
    self.net.cuda()

    # Initialize the model in some way ...
    print("Initializing weights...")
    weights_normal_init(self.net, dev=0.01)
    self.net.load_pretrained_conv(osp.join(globals.data_dir, "VGG_imagenet.npy"))

    # Initial object embedding with word2vec
    #with open("../data/vrd/params_emb.pkl") as f:
    #    emb_init = pickle.load(f)
    #net.state_dict()['emb.weight'].copy_(torch.from_numpy(emb_init))

    print("Initializing optimizer...")
    self.criterion = nn.MultiLabelMarginLoss().cuda()
    self.lr = 0.00001
    # self.momentum = 0.9
    self.weight_decay = 0.0005

    # opt_params = list(self.net.parameters())
    opt_params = [
      {'params': self.net.fc8.parameters(),       'lr': self.lr*10},
      {'params': self.net.fc_fusion.parameters(), 'lr': self.lr*10},
      {'params': self.net.fc_rel.parameters(),    'lr': self.lr*10},
    ]
    if(self.net_args.use_so):
      opt_params.append({'params': self.net.fc_so.parameters(), 'lr': self.lr*10})
    if(self.net_args.use_spat == 1):
      opt_params.append({'params': self.net.fc_spatial.parameters(), 'lr': self.lr*10})
    elif(self.net_args.use_spat == 2):
      raise NotImplementedError
      # opt_params.append({'params': self.net.conv_lo.parameters(), 'lr': self.lr*10})
      # opt_params.append({'params': self.net.fc_spatial.parameters(), 'lr': self.lr*10})
    if(self.net_args.use_sem):
      opt_params.append({'params': self.net.fc_semantic.parameters(), 'lr': self.lr*10})
    self.optimizer = torch.optim.Adam(opt_params,
            lr=self.lr,
            # momentum=self.momentum,
            weight_decay=self.weight_decay)

    if load_pretrained:
      model_path = osp.join(globals.models_dir, self.pretrained)

      print("Loading model... (checkpoint {})".format(model_path))

      if not osp.isfile(model_path):
        raise Exception("Pretrained model not found: {}".format(model_path))

      checkpoint = torch.load(model_path)
      self.start_epoch = checkpoint["epoch"]
      # self.session = checkpoint["session"]

      state_dict = checkpoint["state_dict"]
      try:
        self.net.load_state_dict(state_dict)
      except RuntimeError:
        def patch_model_state_dict(state_dict):
          state_dict["fc_semantic.fc.weight"] = state_dict["fc_so_emb.fc.weight"]
          state_dict["fc_semantic.fc.bias"] = state_dict["fc_so_emb.fc.bias"]
          del state_dict["emb.weight"]
          del state_dict["fc_so_emb.fc.weight"]
          del state_dict["fc_so_emb.fc.bias"]
          return state_dict
        self.net.load_state_dict(patch_model_state_dict(state_dict))

      self.optimizer.load_state_dict(checkpoint["optimizer"])

  def train(self):
    res_file = "output-{}.txt".format(self.session_name)

    # headers = ["Epoch","Pre R@50", "ZS", "R@100", "ZS", "Rel R@50", "ZS", "R@100", "ZS"]
    # headers = ["Epoch","Pre R@50", "ZS", "R@100", "ZS"]
    headers = ["Epoch","Pre R@50", "R@100"]
    res = []
    for epoch in range(self.start_epoch, self.start_epoch + self.max_epochs):

      print("Epoch {}".format(epoch))

      self.__train_epoch(epoch)
      # res.append((epoch,) + test_pre_net(net, args) + test_rel_net(net, args))
      res.append((epoch,) + self.test_pre_net())
      with open("results-{}.txt".format(self.session_name), 'w') as f:
        f.write(tabulate(res, headers))

      if epoch % self.checkpoint_frequency == 0:
        save_name = osp.join(self.save_dir, "checkpoint_epoch_{}.pth.tar".format(epoch))
        save_checkpoint({
          "session": self.session_name,
          "epoch": epoch,
          "state_dict": self.net.state_dict(),
          "optimizer_state_dict": self.optimizer.state_dict(),
          # "net_args": net_args
          # "loss": loss,
          # "pooling_mode": cfg.POOLING_MODE,
          # "class_agnostic": self.class_agnostic,
        }, save_name)

  def __train_epoch(self, epoch):
    self.net.train()

    # Obtain next annotation input and target
    #for spatial_features, semantic_features, target in self.datalayer:
    losses = utils.AverageMeter()
    for step in range(self.iters_per_epoch):

      img_blob, \
      obj_boxes, u_boxes, \
      idx_s, idx_o, \
      spatial_features, semantic_features, \
      rel_sop_prior, target = next(self.datalayer)

      # Forward pass & Backpropagation step
      self.optimizer.zero_grad()
      obj_scores, rel_scores = self.net(img_blob, obj_boxes, u_boxes, idx_s, idx_o, spatial_features, semantic_features)

      # applying some preprocessing to the rel_sop_prior before factoring it into the score
      rel_sop_prior = -0.5 * ( rel_sop_prior + 1.0 / self.datalayer.n_pred)
      loss = self.criterion((rel_sop_prior + rel_scores).view(1, -1), target)
      # loss = self.criterion((rel_scores).view(1, -1), target)
      losses.update(loss.item())
      loss.backward()
      self.optimizer.step()

    print("Epoch loss: {}".format(losses.avg))
    losses.reset()

    """
    for step in range(epoch_num):

      # the rel_so_prior here is a subset of the 100*70*70 dimensional so_prior array, which contains the predicate prob distribution for all object pairs
      # the rel_so_prior here contains the predicate probability distribution of only the object pairs in this annotation
      # Also, it might be helpful to keep in mind that this data layer currently works for a single annotation at a time - no batching is implemented!
      image_blob, boxes, rel_boxes, SpatialFea, classes, ix1, ix2, rel_labels, rel_so_prior = self.datalayer.forward()

    """

  # TODO: move in evaluation?
  def test_pre_net(self):
    import numpy as np
    self.net.eval()
    time1 = time.time()

    test_data_layer = VRDDataLayer(self.dataset_args, "test")

    res = {}
    rlp_labels_cell  = []
    tuple_confs_cell = []
    sub_bboxes_cell  = []
    obj_bboxes_cell  = []

    N = 100 # What's this? (num of rel_res) (with this you can compute R@i for any i<=N)

    while True:
      try:
        img_blob, obj_boxes, u_boxes, idx_s, idx_o, spatial_features, semantic_features, classes, ori_bboxes = next(test_data_layer)
      except ValueError:
        break

      tuple_confs_im = np.zeros((N,),   dtype = np.float) # Confidence...
      rlp_labels_im  = np.zeros((N, 3), dtype = np.float) # Rel triples
      sub_bboxes_im  = np.zeros((N, 4), dtype = np.float) # Subj bboxes
      obj_bboxes_im  = np.zeros((N, 4), dtype = np.float) # Obj bboxes

      obj_scores, rel_scores = self.net(img_blob, obj_boxes, u_boxes, idx_s, idx_o, spatial_features, semantic_features)
      rel_prob = rel_scores.data.cpu().numpy()
      rel_res = np.dstack(np.unravel_index(np.argsort(-rel_prob.ravel()), rel_prob.shape))[0][:N]

      for ii in range(rel_res.shape[0]):
        rel = rel_res[ii, 1]
        tuple_idx = rel_res[ii, 0]

        conf = rel_prob[tuple_idx, rel]
        tuple_confs_im[ii] = conf

        rlp_labels_im[ii] = [classes[idx_s[tuple_idx]], rel, classes[idx_o[tuple_idx]]]

        sub_bboxes_im[ii] = ori_bboxes[idx_s[tuple_idx]]
        obj_bboxes_im[ii] = ori_bboxes[idx_o[tuple_idx]]

      # Is this because of the background ... ?
      if(test_data_layer.ds_name == "vrd"):
        rlp_labels_im += 1

      tuple_confs_cell.append(tuple_confs_im)
      rlp_labels_cell.append(rlp_labels_im)
      sub_bboxes_cell.append(sub_bboxes_im)
      obj_bboxes_cell.append(obj_bboxes_im)

    res["rlp_confs_ours"]  = tuple_confs_cell
    res["rlp_labels_ours"] = rlp_labels_cell
    res["sub_bboxes_ours"] = sub_bboxes_cell
    res["obj_bboxes_ours"] = obj_bboxes_cell

    rec_50     = eval_recall_at_N(test_data_layer.ds_name, 50,  res, use_zero_shot = False)
    # rec_50_zs  = eval_recall_at_N(test_data_layer.ds_name, 50,  res, use_zero_shot = True)
    rec_100    = eval_recall_at_N(test_data_layer.ds_name, 100, res, use_zero_shot = False)
    # rec_100_zs = eval_recall_at_N(test_data_layer.ds_name, 100, res, use_zero_shot = True)
    time2 = time.time()

    # print ("CLS TEST r50:%f, r50_zs:%f, r100:%f, r100_zs:%f" % (rec_50, rec_50_zs, rec_100, rec_100_zs))
    print ("CLS TEST r50:%f, r100:%f" % (rec_50, rec_100))
    print ("TEST Time:%s" % (time.strftime('%H:%M:%S', time.gmtime(int(time2 - time1)))))

    # return rec_50, rec_50_zs, rec_100, rec_100_zs
    return rec_50, rec_100

if __name__ == '__main__':
  trainer = vrd_trainer()

  trainer.train()
