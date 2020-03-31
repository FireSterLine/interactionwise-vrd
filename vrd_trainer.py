import os
import os.path as osp
import sys
import pickle
from tabulate import tabulate
import time
import json

import random
# only for debugging (i.e TODO remove)
random.seed(0)
import numpy as np
# only for debugging (i.e TODO remove)
np.random.seed(0)

import torch
# only for debugging (i.e TODO remove)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import torch.nn as nn
import torch.nn.init
import torch.nn.functional as F

from munch import munchify
import globals, utils
from lib.vrd_models import VRDModel
from lib.datalayers import VRDDataLayer
from lib.evaluator import VRDEvaluator
#, save_net, load_net, \
#      adjust_learning_rate, , clip_gradient

class vrd_trainer():

  def __init__(self,
      checkpoint = False,
      args = {
        # Dataset in use
        "data" : {
          "name"      : "vrd",
          "with_bg_obj"  : False,
          "with_bg_pred" : False,
        },
        # Architecture (or model) type
        "model" : {
          # Constructor
          "type" : "DSRModel",
          # In addition to the visual features of the union box,
          #  use those of subject and object individually?
          "use_so" : True,

          # Use visual features
          "use_vis" : True,

          # Use semantic features (TODO: this becomes the size of the semantic features)
          "use_sem" : True,

          # Three types of spatial features:
          # - 0: no spatial info
          # - 1: 8-way relative location vector
          # - 2: dual mask
          "use_spat" : 0,

          # Size of the representation for each modality when fusing features
          "n_fus_neurons" : 256,

          # Use batch normalization or not
          "use_bn" : False,
        },
        # Evaluation Arguments
        "eval" : {
          "use_obj_prior" : True
        },
        # Training parameters
        "training" : {
          "opt" : {
            "lr"           : 0.00001,
            # "momentum"   : 0.9,
            "weight_decay" : 0.0005,
          },

          # Adjust learning rate every lr_decay_step epochs
          # TODO: check if this works:
          #"lr_decay_step"  : 3,
          #"lr_decay_gamma" : .1,

          "use_shuffle"    : True, # TODO: check if shuffle works

          "num_epochs" : 20,
          "checkpoint_freq" : 5,

          # Number of lines printed with loss ...TODO explain smart freq
          "prints_per_epoch" : 10,

          # TODO
          "batch_size" : 1,

          "test_pre" : True,
          "test_rel" : False,
        }
      }):

    # Local Patch:
    # If we run on a CPU, we reduce the computational load for debugging purposes
    if(utils.device == torch.device("cpu")):
      args["training"]["num_epochs"] = 4
      args["data"]["justafew"] = True
      args["training"]["prints_per_epoch"] = 0.3 # 1


    print("Arguments:")
    if checkpoint:
      print("Checkpoint: {}", checkpoint)
    else:
      print("No Checkpoint")
    print("args:", json.dumps(args, indent=2, sort_keys=True))


    self.session_name = "test-new-training"

    self.checkpoint = checkpoint
    self.args       = args
    self.state      = {"epoch" : 0}



    # Load checkpoint, if any
    if isinstance(self.checkpoint, str):

      checkpoint_path = osp.join(globals.models_dir, self.checkpoint)
      print("Loading checkpoint... ({})".format(checkpoint_path))

      if not osp.isfile(checkpoint_path):
        raise Exception("Checkpoint not found: {}".format(checkpoint_path))

      checkpoint = utils.load_checkpoint(checkpoint_path)
      #print(checkpoint.keys())

      # Session name
      utils.patch_key(checkpoint, "session", "session_name") # (patching)
      if "session_name" in checkpoint:
        self.session_name = checkpoint["session_name"]

      # Arguments
      if "args" in checkpoint:
        self.args = checkpoint["args"]

      # State
      # Epoch
      utils.patch_key(checkpoint, "epoch", ["state", "epoch"]) # (patching)
      # Model state dictionary
      utils.patch_key(checkpoint, "state_dict", ["state", "model_state_dict"]) # (patching)
      utils.patch_key(checkpoint["state"]["model_state_dict"], "fc_so_emb.fc.weight", "fc_semantic.fc.weight") # (patching)
      utils.patch_key(checkpoint["state"]["model_state_dict"], "fc_so_emb.fc.bias",   "fc_semantic.fc.bias") # (patching)
      # TODO: is this different than the weights used for initialization...? del checkpoint["model_state_dict"]["emb.weight"]
      # Optimizer state dictionary
      utils.patch_key(checkpoint, "optimizer", ["state", "optimizer_state_dict"]) # (patching)
      self.state = checkpoint["state"]

    # TODO: idea, don't use data_args.name but data.name?
    self.data_args    = munchify(self.args["data"])
    self.model_args   = munchify(self.args["model"])
    self.eval_args    = munchify(self.args["eval"])
    self.training     = munchify(self.args["training"])

    # TODO: change split to avoid overfitting on this split! (https://towardsdatascience.com/understanding-pytorch-with-an-example-a-step-by-step-tutorial-81fc5f8c4e8e)
    #  train_dataset, val_dataset = random_split(dataset, [80, 20])

    # Data
    print("Initializing data...")
    print("Data args: ", self.data_args)
    # TODO: VRDDataLayer has to know what to yield (DRS -> img_blob, obj_boxes, u_boxes, idx_s, idx_o, spatial_features, obj_classes)
    self.datalayer = VRDDataLayer(self.data_args, "train", self.training.use_shuffle)
    self.dataloader = torch.utils.data.DataLoader(
      dataset = self.datalayer,
      batch_size = 1, # 256,
      shuffle = self.training.use_shuffle
    )
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
    self.model = VRDModel(self.model_args).to(device=utils.device)
    if "model_state_dict" in self.state:
      # TODO: Make sure that this doesn't need the random initialization first
      self.model.load_state_dict(self.state["model_state_dict"])
    else:
      # Random initialization
      utils.weights_normal_init(self.model, dev=0.01)
      # Load VGG layers
      self.model.load_pretrained_conv(osp.join(globals.data_dir, "VGG_imagenet.npy"), fix_layers=True)
      # Load existing (word2vec?) embeddings
      with open(osp.join(globals.data_dir, "vrd", "params_emb.pkl"), 'rb') as f:
        self.model.state_dict()["emb.weight"].copy_(torch.from_numpy(pickle.load(f, encoding="latin1")))

    # Evaluation
    print("Initializing evaluation...")
    print("Evaluation args: ", self.eval_args)
    self.eval = VRDEvaluator(self.data_args, self.eval_args)

    # Training
    print("Initializing training...")
    print("Training args: ", self.training)
    self.optimizer = self.model.OriginalAdamOptimizer(**self.training.opt)
    # TODO: create loss_type argument... also, use reduction='sum' instead?
    self.criterion = nn.MultiLabelMarginLoss(reduction="sum").to(utils.device)
    if "optimizer_state_dict" in self.state:
      self.optimizer.load_state_dict(self.state["optimizer_state_dict"])

  # Perform the complete training process
  def train(self):
    print("train()")

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

    end_epoch = self.state["epoch"] + self.training.num_epochs
    while self.state["epoch"] < end_epoch:

      print("Epoch {}/{}".format((self.state["epoch"]+1), end_epoch))


      # TODO check if this works (Note that you'd have to make it work cross-sessions as well)
      # if (self.state["epoch"] % (self.training.lr_decay_step + 1)) == 0:
      #   print("*adjust_learning_rate*")
      #   utils.adjust_learning_rate(self.optimizer, self.training.lr_decay_gamma)
      # TODO do it with the scheduler, see if it's the same: https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate

      # TODO: after the third epoch, we divide learning rate by 3
      # the authors mention doing this in their paper, but we couldn't find it in the actual code
      #if epoch == 2:
      #  print("Dividing the learning rate by 10 at epoch {}!".format(epoch))
      #  for i in range(len(self.optimizer.param_groups)):
      #    self.optimizer.param_groups[i]['lr'] /= 10


      # self.__train_epoch(self.state["epoch"])

      # Test results
      res_row = [self.state["epoch"]]
      if self.training.test_pre:
        rec_50, rec_50_zs, rec_100, rec_100_zs, dtime = self.test_pre()
        res_row += [rec_50, rec_50_zs, rec_100, rec_100_zs]
      if self.training.test_rel:
        rec_50, rec_50_zs, rec_100, rec_100_zs, dtime = self.test_rel()
        res_row += [rec_50, rec_50_zs, rec_100, rec_100_zs]
      res.append(res_row)

      with open(save_file, 'w') as f:
        f.write(tabulate(res, res_headers))

      # Save checkpoint
      if utils.smart_fequency_check(self.state["epoch"], self.training.num_epochs, self.training.checkpoint_freq):

        # TODO: the loss should be a result: self.result.loss (which is ignored at loading,only used when saving checkpoint)...
        self.state["model_state_dict"]     = self.model.state_dict()
        self.state["optimizer_state_dict"] = self.optimizer.state_dict()

        utils.save_checkpoint({
          "session_name"  : self.session_name,
          "args"          : self.args,
          "state"         : self.state,
          "result"        : dict(zip(res_headers, res_row)),
        }, osp.join(save_dir, "checkpoint_epoch_{}.pth.tar".format(self.state["epoch"])))

      self.__train_epoch()

      self.state["epoch"] += 1

  def __train_epoch(self):
    self.model.train()

    time1 = time.time()
    # TODO check if LeveledAverageMeter works
    losses = utils.LeveledAverageMeter(2)

    # Iterate over the dataset
    n_iter = len(self.dataloader)
    
    # for iter in range(n_iter):
    for i_iter,(net_input, rel_soP_prior, target) in enumerate(self.dataloader):

      print("{}/{}".format(i_iter, n_iter))

      # print(type(net_input))
      # print(type(rel_soP_prior))
      # print(type(target))

      batch_size = target.size()[0]

      # Forward pass & Backpropagation step
      self.optimizer.zero_grad()
      _, rel_scores = self.model(*net_input)

      # Preprocessing the rel_soP_prior before factoring it into the loss
      rel_soP_prior.to(utils.device)
      rel_soP_prior = -0.5 * ( rel_soP_prior + (1.0 / self.datalayer.n_pred))

      # TODO: fix this weird target shape in datalayers and remove this view
      loss = self.criterion((rel_soP_prior + rel_scores).view(batch_size, -1), target)
      # loss = self.criterion((rel_scores).view(batch_size, -1), target)

      losses.update(loss.item())
      loss.backward()
      self.optimizer.step()

      # TODO: I'd like to move that thing here, but maybe I can't call item() after backward?
      # losses.update(loss.item())

      if utils.smart_fequency_check(i_iter, n_iter, self.training.prints_per_epoch):
        print("\t{:4d}: LOSS: {: 6.3f}".format(i_iter, losses.avg(0)))
        losses.reset(0)

    self.state["loss"] = losses.avg(1)
    time2 = time.time()

    print("TRAIN Loss: {: 6.3f}".format(self.state["loss"]))
    print("TRAIN Time: {}".format(utils.time_diff_str(time1, time2)))

    """
      # the rel_soP_prior here is a subset of the 100*70*70 dimensional so_prior array, which contains the predicate prob distribution for all object pairs
      # the rel_soP_prior here contains the predicate probability distribution of only the object pairs in this annotation
      # Also, it might be helpful to keep in mind that this data layer currently works for a single annotation at a time - no batching is implemented!
      image_blob, boxes, rel_boxes, SpatialFea, classes, ix1, ix2, rel_labels, rel_soP_prior = self.datalayer...
    """

  def test_pre(self):
    rec_50, rec_50_zs, rec_100, rec_100_zs, dtime = self.eval.test_pre(self.model)
    print("CLS PRED TEST:\nAll:\tR@50: {: 6.3f}\tR@100: {: 6.3f}\nZShot:\tR@50: {: 6.3f}\tR@100: {: 6.3f}".format(rec_50, rec_100, rec_50_zs, rec_100_zs))
    print("TEST Time: {}".format(utils.time_diff_str(dtime)))
    return rec_50, rec_50_zs, rec_100, rec_100_zs, dtime

  def test_rel(self):
    rec_50, rec_50_zs, rec_100, rec_100_zs, pos_num, loc_num, gt_num, dtime = self.eval.test_rel(self.model)
    print("CLS REL TEST:\nAll:\tR@50: {: 6.3f}\tR@100: {: 6.3f}\nZShot:\tR@50: {: 6.3f}\tR@100: {: 6.3f}".format(rec_50, rec_100, rec_50_zs, rec_100_zs))
    print("CLS OBJ TEST POS: {: 6.3f}, LOC: {: 6.3f}, GT: {: 6.3f}, Precision: {: 6.3f}, Recall: {: 6.3f}".format(pos_num, loc_num, gt_num, pos_num/(pos_num+loc_num), pos_num/gt_num))
    print("TEST Time: {}".format(utils.time_diff_str(dtime)))
    return rec_50, rec_50_zs, rec_100, rec_100_zs, dtime

if __name__ == "__main__":
  # trainer = vrd_trainer()
  trainer = vrd_trainer(checkpoint = False)
  # trainer = vrd_trainer(checkpoint = "epoch_4_checkpoint.pth.tar")
  trainer.train()
  #trainer.test_pre()
  # trainer.test_rel()
  #trainer.train()
