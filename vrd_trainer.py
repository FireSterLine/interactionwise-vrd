import os
import os.path as osp
import sys
import pickle
from tabulate import tabulate
import time

import torch
import torch.nn as nn
import torch.nn.init
import torch.nn.functional as F

from bunch import bunchify
import globals, utils
from lib.nets.vrd_model import vrd_model
from lib.datalayers import VRDDataLayer
from lib.evaluation_dsr import eval_recall_at_N, eval_obj_img # TODO remove this module
#, save_net, load_net, \
#      adjust_learning_rate, , clip_gradient

# from lib.model import train_net, test_pre_net, test_rel_net




class DSRAdamOptimizer():

  self.lr = 0.00001
  # self.momentum = 0.9
  self.weight_decay = 0.0005

  # opt_params = list(self.net.parameters())
  opt_params = [
    {'params': self.net.fc8.parameters(),       'lr': self.lr*10},
    {'params': self.net.fc_fusion.parameters(), 'lr': self.lr*10},
    {'params': self.net.fc_rel.parameters(),    'lr': self.lr*10},
  ]
  if(self.model_args.use_so):
    opt_params.append({'params': self.net.fc_so.parameters(), 'lr': self.lr*10})
  if(self.model_args.use_spat == 1):
    opt_params.append({'params': self.net.fc_spatial.parameters(), 'lr': self.lr*10})
  elif(self.model_args.use_spat == 2):
    raise NotImplementedError
    # opt_params.append({'params': self.net.conv_lo.parameters(), 'lr': self.lr*10})
    # opt_params.append({'params': self.net.fc_spatial.parameters(), 'lr': self.lr*10})
  if(self.model_args.use_sem):
    opt_params.append({'params': self.net.fc_semantic.parameters(), 'lr': self.lr*10})

  return torch.optim.Adam(opt_params,
          lr=self.lr,
          # momentum=self.momentum,
          weight_decay=self.weight_decay)






























class vrd_trainer():

  def __init__(self,
      checkpoint = False,
      data_args = {
        "ds_name"      : "vrd",
        "with_bg_obj"  : False,
        "with_bg_pred" : False
      },
      # Architecture (or model) type
      model_type = "dsr-net",
      # The argument dictionary giving shape to the model
      model_args = {
        # In addition to the visual features of the union box,
        #  use those of subject and object individually?
        "use_so" : True,

        # Use visual features
        self.model_args.use_vis = True,

        # Use semantic features (TODO: this becomes the size of the semantic features)
        self.model_args.use_sem = True,

        # Three types of spatial features:
        # - 0: no spatial info
        # - 1: 8-way relative location vector
        # - 2: dual mask
        self.model_args.use_spat = 0,

        # Size of the representation for each modality when fusing features
        self.model_args.n_fus_neurons = 256,

        # Use batch normalization or not
        self.model_args.use_bn = False,
      },
      # Training parameters
      training = {
        "epoch" : 0,
        "num_epochs" : 20,
        "checkpoint_freq" : 5,

        # TODO make this such that -1 means "one full dataset round" (and maybe -2 is two full.., -3 is three but whatevs)
        "iters_per_epoch"  : -1,
        # Number of lines printed with loss ...TODO explain smart freq
        "prints_freq" : 10,

        # TODO
        "batch_size" : 1,

        "test_pre" : True,
        "test_rel" : False,
      }):

    print("vrd_trainer() called with args:")
    print([checkpoint, data_args, model_args, training])

    self.checkpoint      = checkpoint

    self.data_args    = bunchify(data_args)
    self.model_args   = bunchify(model_args)
    self.training     = bunchify(training)

    self.session_name    = "test-new-training"


    # Load checkpoint, if any
    if isinstance(self.checkpoint, str):

      checkpoint_path = osp.join(globals.models_dir, self.checkpoint)
      print("Loading checkpoint... ({})".format(checkpoint_path))

      if not osp.isfile(checkpoint_path):
        raise Exception("Checkpoint not found: {}".format(checkpoint_path))

      checkpoint = torch.load(checkpoint_path)

      # Data arguments
      if not checkpoint.get("data_args") is None:
        self.data_args = checkpoint["data_args"]

      # Model arguments
      if not checkpoint.get("model_args") is None:
        self.model_args   = checkpoint["model_args"]

      # Training arguments
      if not checkpoint.get("training") is None:
        self.training = checkpoint["training"]

      if not checkpoint.get("epoch") is None:          # (patching)
        self.training.epoch = checkpoint["epoch"]


      # Session name
      utils.patch_key(checkpoint, "session", "session_name") # (patching)
      self.session_name = checkpoint["session_name"]


      # Model state dictionary
      utils.patch_key(checkpoint, "state_dict", "model_state_dict") # (patching)
      utils.patch_key(checkpoint["model_state_dict"], "fc_semantic.fc.weight", "fc_so_emb.fc.weight") # (patching)
      utils.patch_key(checkpoint["model_state_dict"], "fc_semantic.fc.bias",   "fc_so_emb.fc.bias") # (patching)
      # TODO: is this different than the weights used for initialization...? del checkpoint["model_state_dict"]["emb.weight"]
      model_state_dict = checkpoint["model_state_dict"]

      # Optimizer state dictionary
      utils.patch_key(checkpoint, "optimizer", "optimizer_state_dict") # (patching)
      optimizer_state_dict = checkpoint["model_state_dict"]
    else:
      # Model state dictionary
      model_state_dict = None
      # Optimizer state dictionary
      optimizer_state_dict = None


    # Data
    print("Initializing data...")
    print("Data args: ", self.data_args)
    self.datalayer = VRDDataLayer(self.data_args, "train")
    # TODO: Pytorch DataLoader instead:
    # self.num_workers = 0
    # self.dataset = VRDDataset()
    # self.datalayer = torch.utils.data.DataLoader(self.dataset,
    #                  batch_size=self.training.batch_size,
    #                  # sampler= Random ...,
    #                  num_workers=self.num_workers)


    # Model
    self.model_args.n_obj  = self.datalayer.n_obj
    self.model_args.n_pred = self.datalayer.n_pred
    print("Initializing VRD Model...")
    print("Model args: ", self.model_args)
    self.net = vrd_model(self.model_args).cuda()
    if not model_state_dict is None:
      # TODO: Make sure that this doesn't need the random initialization first
      self.net.load_state_dict(model_state_dict)
    else:
      # Random initialization
      utils.weights_normal_init(self.net, dev=0.01)
      # Load VGG layers
      self.net.load_pretrained_conv(osp.join(globals.data_dir, "VGG_imagenet.npy"), fix_layers=True)
      # Load existing (word2vec?) embeddings
      with open(osp.join(globals.data_dir, "vrd", "params_emb.pkl", 'rb') as f:
        self.net.state_dict()["emb.weight"].copy_(torch.from_numpy(pickle.load(f, encoding="latin1")))

    # Training
    print("Initializing training...")
    print("Training args: ", self.model_args)
    optimizer...
    self.optimizer = DSRAdamOptimizer(...)
    loss_type...
    self.criterion = nn.MultiLabelMarginLoss().cuda()
    if not optimizer_state_dict is None:
      self.optimizer.load_state_dict(optimizer_state_dict)

    # TODO: evaluation args
    self.use_obj_prior = True

  # Perform the complete training process
  def train(self):
    save_dir = osp.join(globals.models_dir, self.session_name)
    if not osp.exists(save_dir):
      os.mkdir(save_dir)
    save_file = osp.join(globals.models_dir, "{}.txt".format(self.session_name))

    # Prepare result table
    res_headers = ["Epoch"]
    if self.training.test_pre:
      res_headers += ["Pre R@50", "ZS", "R@100", "ZS"]
    if self.training.test_rel:
      res_headers += ["Rel R@50", "ZS", "R@100", "ZS"]
    res = []

    end_epoch = self.training.epoch + self.training.num_epochs
    while self.training.epoch < end_epoch:

      print("Epoch {}".format(self.training.epoch))

      # self.__train_epoch(self.training.epoch)

      # Test results
      res_row = (self.training.epoch,)
      if self.training.test_pre:
        res_row += self.test_pre_net()
      if self.training.test_rel:
        res_row += self.test_rel_net()
      res.append(res_row)

      with open(save_file, 'w') as f:
        f.write(tabulate(res, res_headers))

      # Save checkpoint
      if utils.smart_fequency_check(self.training.epoch, self.training.num_epochs, self.training.checkpoint_freq):
        utils.save_checkpoint({
          "data_args"             : self.data_args,
          "model_args"            : self.model_args,
          "training"              : self.training,

          "session_name"          : self.session_name,
          "model_state_dict"      : self.net.state_dict(),
          "optimizer_state_dict"  : self.optimizer.state_dict(),
          "result"                : dict(zip(res_headers, res_row)),
        }, osp.join(save_dir, "checkpoint_epoch_{}.pth.tar".format(self.training.epoch)))

      self.__train_epoch()

      self.training.epoch += 1

  def __train_epoch(self):
    self.net.train()

    time1 = time.time()
    losses = utils.LeveledAverageMeter(2)

    # Iterate over the dataset
    n_iter = self.training.iters_per_epoch
    if n_iter < 0:
      # TODO: check that vrd during training ignores None images
      n_iter = self.datalayer.N * (-n_iter)

    for iter in range(n_iter):

      # Obtain next annotation input and target
      net_input, rel_sop_prior, target = self.datalayer.next()

      # TODO: is this necessary? If it's what I think, the datalayer should already ignore these
      if target.size()[1] == 0:
        continue

      # Forward pass & Backpropagation step
      self.optimizer.zero_grad()
      obj_scores, rel_scores = self.net(*net_input)

      # Preprocessing the rel_sop_prior before factoring it into the loss
      rel_sop_prior = torch.FloatTensor(rel_sop_prior).cuda()
      rel_sop_prior = -0.5 * ( rel_sop_prior + 1.0 / self.datalayer.n_pred)
      loss = self.criterion((rel_sop_prior + rel_scores).view(1, -1), target)
      # loss = self.criterion((rel_scores).view(1, -1), target)

      losses.update(loss.item())
      loss.backward()
      self.optimizer.step()

      # TODO: I'd like to move that thing here, but maybe I can't call item() after backward?
      # losses.update(loss.item())

      if utils.smart_fequency_check(step, n_iter, self.training.prints_per_epoch):
        print("\t{:4d}: LOSS: {: 6.3f}".format(step, losses.avg(0)))
        losses.reset(0)

    self.training.loss = losses.avg(1)
    time2 = time.time()

    print("TRAIN Loss: {: 6.3f}".format(self.training.loss))
    print("TRAIN Time: {}".format(utils.time_diff_str(time1, time2)))

    """
    for iter in range(epoch_num):

      # the rel_sop_prior here is a subset of the 100*70*70 dimensional so_prior array, which contains the predicate prob distribution for all object pairs
      # the rel_sop_prior here contains the predicate probability distribution of only the object pairs in this annotation
      # Also, it might be helpful to keep in mind that this data layer currently works for a single annotation at a time - no batching is implemented!
      image_blob, boxes, rel_boxes, SpatialFea, classes, ix1, ix2, rel_labels, rel_sop_prior = self.datalayer.forward()
    """

  # TODO: move in evaluation?
  def test_pre_net(self):
    import numpy as np
    self.net.eval()
    time1 = time.time()

    # TODO: just one VRD test layer
    test_data_layer = VRDDataLayer(self.data_args, "test")

    res = {}
    rlp_labels_cell  = []
    tuple_confs_cell = []
    sub_bboxes_cell  = []
    obj_bboxes_cell  = []

    N = 100 # What's this? (num of rel_res) (with this you can compute R@i for any i<=N)

    while True:

      try:
        net_input, \
          obj_classes_out, ori_bboxes = test_data_layer.next()
      except StopIteration:
        print("StopIteration")
        break

      if(net_input is None):
        rlp_labels_cell.append(None)
        tuple_confs_cell.append(None)
        sub_bboxes_cell.append(None)
        obj_bboxes_cell.append(None)
        continue

      img_blob, obj_boxes, u_boxes, idx_s, idx_o, spatial_features, obj_classes = net_input

      tuple_confs_im = np.zeros((N,),   dtype = np.float) # Confidence...
      rlp_labels_im  = np.zeros((N, 3), dtype = np.float) # Rel triples
      sub_bboxes_im  = np.zeros((N, 4), dtype = np.float) # Subj bboxes
      obj_bboxes_im  = np.zeros((N, 4), dtype = np.float) # Obj bboxes

      obj_scores, rel_scores = self.net(*net_input)
      rel_prob = rel_scores.data.cpu().numpy()
      rel_res = np.dstack(np.unravel_index(np.argsort(-rel_prob.ravel()), rel_prob.shape))[0][:N]

      for ii in range(rel_res.shape[0]):
        rel = rel_res[ii, 1]
        tuple_idx = rel_res[ii, 0]

        conf = rel_prob[tuple_idx, rel]
        tuple_confs_im[ii] = conf

        rlp_labels_im[ii] = [obj_classes_out[idx_s[tuple_idx]], rel, obj_classes_out[idx_o[tuple_idx]]]

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
    rec_50_zs  = eval_recall_at_N(test_data_layer.ds_name, 50,  res, use_zero_shot = True)
    rec_100    = eval_recall_at_N(test_data_layer.ds_name, 100, res, use_zero_shot = False)
    rec_100_zs = eval_recall_at_N(test_data_layer.ds_name, 100, res, use_zero_shot = True)
    time2 = time.time()

    print("CLS PRED TEST:\nAll:\tR@50: {: 6.3f}\tR@100: {: 6.3f}\nZShot:\tR@50: {: 6.3f}\tR@100: {: 6.3f}".format(rec_50, rec_100, rec_50_zs, rec_100_zs))
    print("TEST Time: {}".format(utils.time_diff_str(time1, time2)))

    return rec_50, rec_50_zs, rec_100, rec_100_zs

  def test_rel_net(self):
      import numpy as np
      self.net.eval()
      time1 = time.time()

      test_data_layer = VRDDataLayer(self.data_args.ds_name, "test")

      with open("data/{}/test.pkl".format(self.data_args.ds_name), 'rb') as fid:
        anno = pickle.load(fid, encoding="latin1")

      # TODO: proposals is not ordered, but a dictionary with im_path keys
      # TODO: expand so that we don't need the proposals pickle, and we generate it if it's not there, using Faster-RCNN?
      # TODO: move the proposals file path to a different one (maybe in Faster-RCNN)
      with open("data/{}/proposal.pkl".format(self.data_args.ds_name), 'rb') as fid:
        proposals = pickle.load(fid, encoding="latin1")
        # TODO: zip these
        pred_boxes   = proposals["boxes"]
        pred_classes = proposals["cls"]
        pred_confs   = proposals["confs"]

      N = 100 # What's this? (num of rel_res) (with this you can compute R@i for any i<=N)

      pos_num = 0.0
      loc_num = 0.0
      gt_num  = 0.0

      res = {}
      rlp_labels_cell  = []
      tuple_confs_cell = []
      sub_bboxes_cell  = []
      obj_bboxes_cell  = []
      predict = []

      if len(anno) != len(proposals["cls"]):
        print("ERROR: something is wrong in prediction: {} != {}".format(len(anno), len(proposals["cls"])))

      for step,anno_img in enumerate(anno):

        objdet_res = {"boxes"   : pred_boxes[step], \
                      "classes" : pred_classes[step].reshape(-1), \
                      "confs"   : pred_confs[step].reshape(-1)
                      }

        try:
          net_input, \
            obj_classes_out, rel_sop_prior, ori_bboxes = test_data_layer.next(objdet_res)
        except StopIteration:
          print("StopIteration")
          break

        if(net_input is None):
          rlp_labels_cell.append(None)
          tuple_confs_cell.append(None)
          sub_bboxes_cell.append(None)
          obj_bboxes_cell.append(None)
          continue

        gt_boxes = anno_img["boxes"].astype(np.float32)
        gt_cls = np.array(anno_img["classes"]).astype(np.float32)

        obj_score, rel_score = self.net(*net_input) # img_blob, obj_boxes, u_boxes, idx_s, idx_o, spatial_features, obj_classes)

        _, obj_pred = obj_score[:, 1::].data.topk(1, 1, True, True)
        obj_score = F.softmax(obj_score, dim=1)[:, 1::].data.cpu().numpy()

        rel_prob = rel_score.data.cpu().numpy()
        rel_prob += np.log(0.5*(rel_sop_prior+1.0 / test_data_layer.n_pred))

        img_blob, obj_boxes, u_boxes, idx_s, idx_o, spatial_features, obj_classes = net_input

        pos_num_img, loc_num_img = eval_obj_img(gt_boxes, gt_cls, ori_bboxes, obj_pred.cpu().numpy(), gt_thr=0.5)
        gt_num += gt_boxes.shape[0]
        pos_num += pos_num_img
        loc_num += loc_num_img

        tuple_confs_im = []
        rlp_labels_im  = np.zeros((rel_prob.shape[0]*rel_prob.shape[1], 3), dtype = np.float)
        sub_bboxes_im  = np.zeros((rel_prob.shape[0]*rel_prob.shape[1], 4), dtype = np.float)
        obj_bboxes_im  = np.zeros((rel_prob.shape[0]*rel_prob.shape[1], 4), dtype = np.float)
        n_idx = 0

        for tuple_idx in range(rel_prob.shape[0]):
          for rel in range(rel_prob.shape[1]):
            if(self.use_obj_prior):
              if(objdet_res["confs"].ndim == 1):
                conf = np.log(objdet_res["confs"][idx_s[tuple_idx]]) + np.log(objdet_res["confs"][idx_o[tuple_idx]]) + rel_prob[tuple_idx, rel]
              else:
                conf = np.log(objdet_res["confs"][idx_s[tuple_idx], 0]) + np.log(objdet_res["confs"][idx_o[tuple_idx], 0]) + rel_prob[tuple_idx, rel]
            else:
              conf = rel_prob[tuple_idx, rel]
            tuple_confs_im.append(conf)
            sub_bboxes_im[n_idx] = ori_bboxes[idx_s[tuple_idx]]
            obj_bboxes_im[n_idx] = ori_bboxes[idx_o[tuple_idx]]
            rlp_labels_im[n_idx] = [obj_classes_out[idx_s[tuple_idx]], rel, obj_classes_out[idx_o[tuple_idx]]]
            n_idx += 1

        # Is this because of the background ... ?
        if(self.data_args.ds_name == "vrd"):
          rlp_labels_im += 1

        # Why is this needed? ...
        tuple_confs_im = np.array(tuple_confs_im)
        idx_order = tuple_confs_im.argsort()[::-1][:N]
        rlp_labels_im = rlp_labels_im[idx_order,:]
        tuple_confs_im = tuple_confs_im[idx_order]
        sub_bboxes_im  = sub_bboxes_im[idx_order,:]
        obj_bboxes_im  = obj_bboxes_im[idx_order,:]

        rlp_labels_cell.append(rlp_labels_im)
        tuple_confs_cell.append(tuple_confs_im)
        sub_bboxes_cell.append(sub_bboxes_im)
        obj_bboxes_cell.append(obj_bboxes_im)

        step += 1

      res["rlp_confs_ours"]  = tuple_confs_cell
      res["rlp_labels_ours"] = rlp_labels_cell
      res["sub_bboxes_ours"] = sub_bboxes_cell
      res["obj_bboxes_ours"] = obj_bboxes_cell

      if len(anno) != len(res["obj_bboxes_ours"]):
        print("ERROR: something is wrong in prediction: {} != {}".format(len(anno), len(res["obj_bboxes_ours"])))

      rec_50     = eval_recall_at_N(test_data_layer.ds_name, 50,  res, use_zero_shot = False)
      rec_50_zs  = eval_recall_at_N(test_data_layer.ds_name, 50,  res, use_zero_shot = True)
      rec_100    = eval_recall_at_N(test_data_layer.ds_name, 100, res, use_zero_shot = False)
      rec_100_zs = eval_recall_at_N(test_data_layer.ds_name, 100, res, use_zero_shot = True)
      time2 = time.time()

      print("CLS OBJ TEST POS: {: 6.3f}, LOC: {: 6.3f}, GT: {: 6.3f}, Precision: {: 6.3f}, Recall: {: 6.3f}".format(pos_num, loc_num, gt_num, pos_num/(pos_num+loc_num), pos_num/gt_num))
      print("CLS REL TEST:\nAll:\tR@50: {: 6.3f}\tR@100: {: 6.3f}\nZShot:\tR@50: {: 6.3f}\tR@100: {: 6.3f}".format(rec_50, rec_100, rec_50_zs, rec_100_zs))
      print("TEST Time: {}".format(utils.time_diff_str(time1, time2)))

      return rec_50, rec_50_zs, rec_100, rec_100_zs

if __name__ == '__main__':
  trainer = vrd_trainer()
  # trainer = vrd_trainer(checkpoint = False, self.data_args.ds_name = "vrd")
  # trainer = vrd_trainer(checkpoint = "epoch_4_checkpoint.pth.tar")

  trainer.train()
