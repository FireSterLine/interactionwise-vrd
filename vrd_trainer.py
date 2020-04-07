import os
import os.path as osp
import sys
import pickle
from tabulate import tabulate
import time
import json
import warnings

#import random
# only for debugging (i.e TODO remove)
#random.seed(0)
import numpy as np
# only for debugging (i.e TODO remove)
np.random.seed(0)

import torch
# only for debugging (i.e TODO remove)
torch.manual_seed(0)
#torch.backends.cudnn.deterministic = True
#torch.backends.cudnn.benchmark = False
torch.backends.cudnn.benchmark = True

import torch.nn as nn
import torch.nn.init
import torch.nn.functional as F

from munch import munchify
import globals, utils
from lib.vrd_models import VRDModel
from lib.datalayers import VRDDataLayer
from lib.evaluator import VRDEvaluator

TESTOVERFIT = False # True
TESTVALIDITY = False # True # False # True
DEBUGGING = True # False # True # False

if utils.device == torch.device("cpu"):
  DEBUGGING = True

if TESTOVERFIT: DEBUGGING = False
if TESTOVERFIT: TESTVALIDITY = False
if TESTVALIDITY: DEBUGGING = False

class vrd_trainer():

  def __init__(self, session_name, args = {}, profile = None, checkpoint = False):

    def_args = utils.cfg_from_file("cfgs/default.yml")
    if profile is not None:
      def_args = utils.dict_patch(utils.cfg_from_file(profile), def_args)
    args = utils.dict_patch(args, def_args)

    if DEBUGGING:
      # args["training"]["num_epochs"] = 6
      args["data"]["justafew"] = True
      args["training"]["use_shuffle"] = False
      torch.backends.cudnn.benchmark = False
    if TESTOVERFIT:
      args["data"]["justafew"] = 3
      args["eval"]["justafew"] = 3
    if TESTVALIDITY:
      args["data"]["name"] = "vrd/dsr"
      args["training"]["print_freq"] = 0.1
      args["model"]["use_pred_false"] = True
      # args["training"]["use_preload"] = False


    print("Arguments:")
    if checkpoint:
      print("Checkpoint: {}", checkpoint)
    else:
      print("No Checkpoint")
    print("args:", json.dumps(args, indent=2, sort_keys=True))


    self.session_name = session_name

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
      # TODO: is checkpoint["model_state_dict"]["emb.weight"] different from the weights used for initialization...?
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
    print("Initializing data: ", self.data_args)
    # TODO: VRDDataLayer has to know what to yield (DRS -> img_blob, obj_boxes, u_boxes, idx_s, idx_o, spatial_features, obj_classes)
    self.datalayer = VRDDataLayer(self.data_args, "train", use_preload = self.training.use_preload)
    self.dataloader = torch.utils.data.DataLoader(
      dataset = self.datalayer,
      batch_size = 1, # self.training.batch_size,
      # sampler= Random ...,
      # num_workers=self.num_workers
      shuffle = self.training.use_shuffle,
    )

    # Model
    self.model_args.n_obj  = self.datalayer.n_obj
    self.model_args.n_pred = self.datalayer.n_pred
    if self.model_args.use_pred_sem == True:
      self.model_args.pred_emb = np.array(self.datalayer.dataset.readJSON("predicates-emb.json"))
    print("Initializing VRD Model: ", self.model_args)
    self.model = VRDModel(self.model_args).to(utils.device)
    if "model_state_dict" in self.state:
      # TODO: Make sure that this doesn't need the random initialization first
      print("Loading state_dict")
      self.model.load_state_dict(self.state["model_state_dict"])
    else:
      print("Random initialization")
      # Random initialization
      utils.weights_normal_init(self.model, dev=0.01)
      # Load VGG layers
      self.model.load_pretrained_conv(osp.join(globals.data_dir, "VGG_imagenet.npy"), fix_layers=True)
      # Load existing embeddings
      try:
        with open(osp.join(self.datalayer.dataset.metadata_dir, "params_emb.pkl"), 'rb') as f:
          self.model.state_dict()["emb.weight"].copy_(torch.from_numpy(pickle.load(f, encoding="latin1")))
      except FileNotFoundError:
        warnings.warn("Initialization weights for emb.weight layer not found!", UserWarning)
    # Evaluation
    print("Initializing evaluator: ", self.eval_args)
    self.eval = VRDEvaluator(self.data_args, self.eval_args)

    # Training
    print("Initializing training: ", self.training)
    self.optimizer = self.model.OriginalAdamOptimizer(**self.training.opt)

    if self.training.loss == "mlab":
      self.criterion = nn.MultiLabelMarginLoss(reduction="sum").to(device=utils.device)
    elif self.training.loss == "cross-entropy":
      self.criterion = nn.CrossEntropyLoss(reduction="sum").to(device=utils.device)
    elif self.training.loss == "mse":
      self.criterion = nn.MSELoss(reduction="sum").to(device=utils.device)
    else:
      raise ValueError("Unknown loss specified: '{}'".format(self.training.loss))

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
    if self.eval_args.test_pre:
      res_headers += ["Pre R4x", "ZS", "R@50", "ZS", "R@100", "ZS"]
    if self.eval_args.test_rel:
      res_headers += ["Rel R@4x", "ZS", "R@50", "ZS", "R@100", "ZS"]
    res = []

    end_epoch = self.state["epoch"] + self.training.num_epochs
    while self.state["epoch"] < end_epoch:

      # print("Epoch {}/{}".format((self.state["epoch"]+1), end_epoch))


      # TODO check if this works (Note that you'd have to make it work cross-sessions as well)
      # if (self.state["epoch"] % (self.training.lr_decay_step + 1)) == 0:
      #   print("*adjust_learning_rate*")
      #   utils.adjust_learning_rate(self.optimizer, self.training.lr_decay_gamma)
      # TODO do it with the scheduler, see if it's the same: https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
      # exp_lr_scheduler = lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.1)

      # TODO: after the third epoch, we divide learning rate by 10
      # the authors mention doing this in their paper, but we couldn't find it in the actual code
      if self.state["epoch"] != 0 and (self.state["epoch"] % 3) == 0:
        print("Dividing the learning rate by 10 at epoch {}!".format(self.state["epoch"]))
        for i in range(len(self.optimizer.param_groups)):
          self.optimizer.param_groups[i]["lr"] /= 10

      # self.__train_epoch()

      # Test results
      res_row = [self.state["epoch"]]
      if self.eval_args.test_pre:
        rec_4x, rec_4x_zs, rec_50, rec_50_zs, rec_100, rec_100_zs, dtime = self.test_pre()
        res_row += [rec_4x, rec_4x_zs, rec_50, rec_50_zs, rec_100, rec_100_zs]
      if self.eval_args.test_rel:
        rec_4x, rec_4x_zs, rec_50, rec_50_zs, rec_100, rec_100_zs, dtime = self.test_rel()
        res_row += [rec_4x, rec_4x_zs, rec_50, rec_50_zs, rec_100, rec_100_zs]
      res.append(res_row)

      with open(save_file, 'w') as f:
        f.write(tabulate(res, res_headers))

      # Save checkpoint
      if utils.smart_frequency_check(self.state["epoch"], self.training.num_epochs, self.training.checkpoint_freq):

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
    for i_iter,(net_input, gt_soP_prior, gt_pred_sem, mlab_target) in enumerate(self.dataloader):

      # print("{}/{}".format(i_iter, n_iter))

      # print(type(net_input))
      # print(type(gt_soP_prior))
      # print(type(mlab_target))

      batch_size = mlab_target.size()[0]

      # Forward pass & Backpropagation step
      self.optimizer.zero_grad()
      model_output = self.model(*net_input)

      # Preprocessing the gt_soP_prior before factoring it into the loss
      gt_soP_prior = gt_soP_prior.to(utils.device)
      # Note that maybe introducing no_predicate may be better:
      #  After all, there may not be a relationship between two objects...
      #  And this would avoid dirtying up the predictions?
      gt_soP_prior = -0.5 * ( gt_soP_prior + (1.0 / self.datalayer.n_pred))

      # DSR:
      # TODO: fix this weird-shaped mlab_target in datalayers and remove this view thingy
      if self.training.loss == "mlab":
        _, rel_scores = model_output
        loss = self.criterion((gt_soP_prior + rel_scores).view(batch_size, -1), mlab_target)
        # loss = self.criterion((rel_scores).view(batch_size, -1), mlab_target)
      elif self.training.loss == "mse":
        _, pred_sem = model_output
        # TODO use the weighted embeddings of gt_soP_prior ?
        loss = self.criterion(pred_sem, gt_pred_sem)

      loss.backward()
      self.optimizer.step()

      # Track loss
      losses.update(loss.item())

      if utils.smart_frequency_check(i_iter, n_iter, self.training.print_freq):
          print("\t{:4d}/{:<4d}: LOSS: {: 6.3f}\r".format(i_iter, n_iter, losses.avg(0)), end="")
          losses.reset(0)

    self.state["loss"] = losses.avg(1)
    time2 = time.time()

    print("TRAIN Loss: {: 6.3f}".format(self.state["loss"]))
    print("TRAIN Time: {}".format(utils.time_diff_str(time1, time2)))

    """
      # the gt_soP_prior here is a subset of the 100*70*70 dimensional so_prior array, which contains the predicate prob distribution for all object pairs
      # the gt_soP_prior here contains the predicate probability distribution of only the object pairs in this annotation
      # Also, it might be helpful to keep in mind that this data layer currently works for a single annotation at a time - no batching is implemented!
      image_blob, boxes, rel_boxes, SpatialFea, classes, ix1, ix2, rel_labels, gt_soP_prior = self.datalayer...
    """

  def test_pre(self):
    (rec_100, rec_100_zs, rec_50, rec_50_zs, rel_4x, rel_4x_zs), dtime = self.eval.test_pre(self.model, [100, 50, 4.])
    print("CLS PRED TEST:\nAll:\tR@4x: {: 6.3f}\tR@50: {: 6.3f}\tR@100: {: 6.3f}\nZShot:\tR@4x: {: 6.3f}\tR@50: {: 6.3f}\tR@100: {: 6.3f}".format(rec_4x, rec_50, rec_100, rec_4x_zs, rec_50_zs, rec_100_zs))
    print("TEST Time: {}".format(utils.time_diff_str(dtime)))
    return rec_4x, rec_4x_zs, rec_50, rec_50_zs, rec_100, rec_100_zs, dtime

  def test_rel(self):
    (rec_100, rec_100_zs, rec_50, rec_50_zs, rel_4x, rel_4x_zs), pos_num, loc_num, gt_num, dtime = self.eval.test_rel(self.model, [100, 50, 4.])
    print("CLS REL TEST:\nAll:\tR@4x: {: 6.3f}\tR@50: {: 6.3f}\tR@100: {: 6.3f}\nZShot:\tR@4x: {: 6.3f}\tR@50: {: 6.3f}\tR@100: {: 6.3f}".format(rec_4x, rec_50, rec_100, rec_4x_zs, rec_50_zs, rec_100_zs))
    print("CLS OBJ TEST POS: {: 6.3f}, LOC: {: 6.3f}, GT: {: 6.3f}, Precision: {: 6.3f}, Recall: {: 6.3f}".format(pos_num, loc_num, gt_num, np.float64(pos_num)/(pos_num+loc_num), np.float64(pos_num)/gt_num))
    print("TEST Time: {}".format(utils.time_diff_str(dtime)))
    return rec_4x, rec_4x_zs, rec_50, rec_50_zs, rec_100, rec_100_zs, dtime

if __name__ == "__main__":
  trainer = vrd_trainer("original-checkpoint", checkpoint="epoch_4_checkpoint.pth.tar")
  #trainer = vrd_trainer("original")
  #trainer = vrd_trainer("test", {"training" : {"num_epochs" : 1}, "eval" : {"test_pre" : True, "test_rel" : True}}, checkpoint = False)
  trainer.train()
  sys.exit(0)
  for lr in [0.001, 0.00001]: # [0.001, 0.0001, 0.00001, 0.000001]:
    for weight_decay in [0.0005]:
        for lr_rel_fus_ratio in [1]: # , 10, 100]:
          trainer = vrd_trainer("pred-sem-scan-2-{}-{}-{}".format(lr, weight_decay, lr_rel_fus_ratio), {"training" : {"opt": {"lr": lr, "weight_decay" : weight_decay, "lr_fus_ratio" : lr_rel_fus_ratio, "lr_rel_ratio" : lr_rel_fus_ratio},"checkpoint_freq" : 0.1 },}, profile = "cfgs/pred_sem.yml", checkpoint = False)
          trainer.train()

  # trainer = vrd_trainer({}, checkpoint = "epoch_4_checkpoint.pth.tar")
  #trainer.train()
  #trainer.test_pre()
  # trainer.test_rel()
  #trainer.train()
