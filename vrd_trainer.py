import os
import os.path as osp
import sys
from datetime import datetime
import time

import pickle
import yaml
from munch import munchify, unmunchify
from tabulate import tabulate
from copy import deepcopy

import random
random.seed(0)
import numpy as np
np.random.seed(0)
import torch
torch.manual_seed(0)

import globals, utils
import pandas as pd

# VRD Model, Dataset, Datalayer, Evaluator
from lib.vrd_models import VRDModel
from lib.dataset import VRDDataset
from lib.datalayer import VRDDataLayer, net_input_to, loss_targets_to
from lib.evaluator import VRDEvaluator

#####################################################################################
# Test if code compiles
TEST_DEBUGGING = False
# Test if a newly-introduced change affects the validity of the code
TEST_EVAL_VALIDITY = False
TEST_TRAIN_VALIDITY = False
# Test to overfit to a single element
TEST_OVERFIT = False

FEATURES_SCAN = False
PARAMS_SCAN = True
#####################################################################################

# Fallback for local run
if utils.device == torch.device("cpu"):
  DEBUGGING = True

# A VRDTrainer trainer class initializes the necessary elements for performing a training session,
#  and provides a .train() method for executing the training procedure for the specified numbers of epochs
class VRDTrainer():

  # A training session is initialized with:
  #  # A "session_name", used as an identifier when saving the results
  #  # A dictionary "args" used for specifying the training options.
  #  # A "profile" (string or list of strings) specitying the name of a pre-determined set of training options to load
  #  # A "checkpoint" file to load (False or string indicating the filepath)
  #
  #  The default options are specified in the "cfgs/default.yml" configuration file, but they can be overridden
  #   by a loaded profile (e.g "only_sem", which uses the options in "cfgs/only_sem.yml" to override the default ones)
  #   Furthermore, the set of training options can be overridden again with the specified "args" dictionary.
  #   The "priority list" for determining the training options is: args > profile > "cfgs/default.yml"
  def __init__(self, session_name, args = {}, profile = None, checkpoint = False):

    # Load arguments cascade from profiles
    default_args = utils.load_cfg_profile("default")
    if profile is not None:
      profile = utils.listify(profile)
      for p in profile:
        default_args = utils.dict_patch(utils.load_cfg_profile(p), default_args)
    args = utils.dict_patch(args, default_args)

    #print("Arguments:\n", yaml.dump(args, default_flow_style=False))

    self.session_name = session_name
    self.args         = args
    self.checkpoint   = checkpoint

    self.state        = {"epoch" : 0}

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

    print("Arguments:\n", yaml.dump(self.args, default_flow_style=False))
    self.args = munchify(self.args)

    # TODO: change split to avoid overfitting on this split! (https://towardsdatascience.com/understanding-pytorch-with-an-example-a-step-by-step-tutorial-81fc5f8c4e8e)
    #  train_dataset, val_dataset = random_split(dataset, [80, 20])

    # Data
    print("Initializing data: ", self.args.data)
    self.dataset = VRDDataset(**self.args.data)

    # Model
    self.args.model.n_obj     = self.dataset.n_obj
    self.args.model.n_pred    = self.dataset.n_pred
    self.args.model.emb_size  = globals.emb_model_size(self.dataset.emb_model)
    if self.args.model.use_pred_sem:
      self.args.model.pred_emb = self.dataset.getEmb("predicates")
    print("Initializing VRD Model: ", self.args.model)
    self.model = VRDModel(self.args.model).to(utils.device)
    if "model_state_dict" in self.state:
      print("Loading state_dict")
      self.model.load_state_dict(self.state["model_state_dict"])
    else:
      print("Random initialization...")
      # Random initialization
      utils.weights_normal_init(self.model, dev=0.01)
      # Load existing embeddings
      if self.model.args.feat_used.sem:
        obj_emb = torch.from_numpy(self.dataset.getEmb("objects"))
        self.model.state_dict()["emb.weight"].copy_(obj_emb)
        #TODO: try set_trainability(self.model.emb, requires_grad=True)

    # Training
    print("Initializing training...")
    self.optimizer = self.model.OriginalAdamOptimizer(**self.args.opt)

    self.criterions = {}
    y_cols = []
    if "mlab" in self.args.training.loss:
      y_cols.append("mlab")
      self.criterions["mlab"] = torch.nn.MultiLabelMarginLoss(reduction="none").to(device=utils.device)
    if "bcel" in self.args.training.loss:
      y_cols.append("1-hots")
      self.criterions["bcel"] = torch.nn.BCEWithLogitsLoss(reduction="none").to(device=utils.device)
    if "mse" in self.args.training.loss:
      self.criterions["mse"] = torch.nn.MSELoss(reduction="none").to(device=utils.device)
    if "cross-entropy" in self.args.training.loss:
      # Warning, this doesn't work/do well for multi-label classification
      self.criterions["cross-entropy"] = torch.nn.CrossEntropyLoss(reduction="none").to(device=utils.device)
    if not len(self.criterions):
      raise ValueError("Unknown loss specified: '{}'".format(self.args.training.loss))

    if "optimizer_state_dict" in self.state:
      print("Loading optimizer state_dict...")
      self.optimizer.load_state_dict(self.state["optimizer_state_dict"])
    elif isinstance(self.checkpoint, str):
      print("Warning! Optimizer state_dict not found!")

    # DataLoader
    self.datalayer = VRDDataLayer(self.dataset, "train", use_preload = self.args.training.use_preload, x_cols = self.model.x_cols, y_cols = y_cols)
    self.dataloader = torch.utils.data.DataLoader(
      dataset = self.datalayer,
      batch_size = 1, # self.args.training.batch_size,
      # sampler= Random ...,
      num_workers = 1,
      pin_memory = True,
      shuffle = self.args.training.use_shuffle,
    )

    # Evaluation
    print("Initializing evaluator...")
    self.args.eval.rec = sorted(self.args.eval.rec, reverse=True)
    self.eval = VRDEvaluator(self.dataset, self.args.eval, x_cols = self.model.x_cols)


  # Perform the complete training process
  def train(self, output_results = False):
    print("train()")

    # Prepare result table
    res = []
    res_headers = ["Epoch"]
    if self.args.eval.test_pre: res_headers += self.gt_headers(self.args.eval.test_pre, "Pre") + ["Avg"]
    if self.args.eval.test_rel: res_headers += self.gt_headers(self.args.eval.test_rel, "Rel") + ["Avg"]
    if self.args.eval.by_predicates: # TODO: fix the two (gt+gt_zs)
      if self.args.eval.test_pre:
        for i in range(len(self.args.eval.rec)*((self.eval.gt    is not None) + (self.eval.gt    is not None))):
          res_headers += [x if i else "Pre "+x for i,x in enumerate(self.dataset.pred_classes)]
        res_headers += [x if i else "Pre avg. "+x for i,x in enumerate(self.dataset.pred_classes)]
      if self.args.eval.test_rel:
        for i in range(len(self.args.eval.rec)*((self.eval.gt_zs is not None) + (self.eval.gt_zs is not None))):
          res_headers += [x if i else "Rel "+x for i,x in enumerate(self.dataset.pred_classes)]
        res_headers += [x if i else "Rel avg. "+x for i,x in enumerate(self.dataset.pred_classes)]

    # Test before training?
    if self.args.training.test_first:
      self.do_test(res, res_headers, self.state["epoch"])

    # Start training
    end_epoch = self.state["epoch"] + self.args.training.num_epochs
    while self.state["epoch"] < end_epoch:

      print("Epoch {}/{}".format((self.state["epoch"]+1), end_epoch))


      # TODO check if this works (Note that you'd have to make it work cross-sessions as well)
      # if (self.state["epoch"] % (self.args.training.lr_decay_step + 1)) == 0:
      #   print("*adjust_learning_rate*")
      #   utils.adjust_learning_rate(self.optimizer, self.args.training.lr_decay_gamma)
      # TODO do it with the scheduler, see if it's the same: https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
      # exp_lr_scheduler = lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.1)

      # After the third epoch, we divide learning rate by 10
      #  (the authors mention doing this in their paper, but we couldn't find it in the actual code)
      if self.state["epoch"] == 3:
        print("Dividing the learning rate by 10 at epoch {}!".format(self.state["epoch"]))
        for i in range(len(self.optimizer.param_groups)):
          self.optimizer.param_groups[i]["lr"] /= 10

      # Train
      self.__train_epoch()

      # Save test results and save checkpoint
      is_last_epoch = ((end_epoch-self.state["epoch"]) <= 1)
      do_save_checkpoint = (is_last_epoch and float(self.args.training.checkpoint_freq) != 0.0) or utils.smart_frequency_check(self.state["epoch"], end_epoch, self.args.training.checkpoint_freq, last = True)
      do_test = is_last_epoch or do_save_checkpoint or utils.smart_frequency_check(self.state["epoch"], end_epoch, self.args.training.test_freq, last = True)

      if do_test:
        self.do_test(res, res_headers)

      if do_save_checkpoint:

        # Target directory
        save_dir = osp.join(globals.models_dir, self.session_name)
        if not osp.exists(save_dir):
          os.mkdir(save_dir)

        # Save state
        # TODO: the loss should be a result: self.result.loss (which is ignored at loading,only used when saving checkpoint)...
        self.state["model_state_dict"]     = self.model.state_dict()
        self.state["optimizer_state_dict"] = self.optimizer.state_dict()

        # Save checkpoint
        utils.save_checkpoint({
          "session_name"  : self.session_name,
          "args"          : unmunchify(self.args),
          "state"         : self.state,
          "result"        : ([res_headers] + res), # dict(zip(res_headers, res[-1])),
        }, osp.join(save_dir, "checkpoint_epoch_{}.pth.tar".format(self.state["epoch"])))

      self.state["epoch"] += 1

    if output_results:
      res_np = np.array(res)
      res_headers_np = np.array(res_headers)

      num_epochs, num_cols = res_np.shape
      res_dict = {}
      res_headers_dict = {}

      # Number of recalls
      res_head_headers = []
      if self.args.eval.test_pre: res_head_headers += self.gt_headers(self.args.eval.test_pre, "Pre") + ["Pre Avg"]
      if self.args.eval.test_rel: res_head_headers += self.gt_headers(self.args.eval.test_rel, "Rel") + ["Rel Avg"]
      num_recalls = len(res_head_headers)

      res_headers_dict["head"] = res_headers_np[:num_recalls+1]
      res_dict["head"]               = res_np[:,:num_recalls+1]

      if self.args.eval.by_predicates:
        res_headers_dict["predicates"] = res_headers_np[[0] + list(range(num_recalls+1,num_cols))]
        res_dict["predicates"]               = res_np[:,[0] + list(range(num_recalls+1,num_cols))]

        predicates_stacked = []
        for i_rec_score,rec_score in enumerate(res_head_headers):
          x = res_dict["predicates"][:,[0] + list(range(1+i_rec_score*self.dataset.n_pred,1+(i_rec_score+1)*self.dataset.n_pred))]
          predicates_stacked += x.tolist()
          predicates_stacked.append([np.nan for _ in range(x.shape[1])])

        res_headers_dict["predicates_stacked"] = np.array(res_headers_dict["predicates"][[0]].tolist() + self.dataset.pred_classes)
        res_dict["predicates_stacked"] = np.array(predicates_stacked)
        
        #res_dict["predicates"]         = res_dict["predicates"].transpose()
        #res_dict["predicates_stacked"] = res_dict["predicates_stacked"].transpose()
        #del(res_dict["predicates"])
        #del(res_headers_dict["predicates"])

      return res_headers_dict, res_dict

  # Test network
  def do_test(self, res, res_headers, force_epoch = None):
    epoch = self.state["epoch"]+1 if force_epoch is None else force_epoch
    res_row = []
    res_row_end = []
    res_row_end_end = []
    # Test Predicate detection and/or Relationship detection
    for do_test,f_test in zip([self.args.eval.test_pre, self.args.eval.test_rel], [self.test_pre, self.test_rel]):
      if do_test:
        if not self.args.eval.by_predicates:
          recalls, dtime = f_test()
        else:
          # If the evaluation also returns the recall scores by predicate, add the by-predicate average columns
          recalls, recalls_by_preds, dtime = f_test()
          res_row_end += recalls_by_preds
          recalls_by_preds_avg = np.zeros(self.dataset.n_pred)
          for rec_score in range(len(recalls)):  # e.g 4
            recalls_by_preds_avg += np.array(recalls_by_preds[rec_score*self.dataset.n_pred:(rec_score+1)*self.dataset.n_pred])
          recalls_by_preds_avg /= len(recalls)
          res_row_end_end += list(recalls_by_preds_avg)
        # Add avg. of recall scores
        res_row += recalls + (sum(recalls)/len(recalls),)

    # Update results
    res.append([epoch] + res_row + res_row_end + res_row_end_end)

    # Save results to file
    with open(osp.join(globals.models_dir, "{}.txt".format(self.session_name)), 'w') as f:
      f.write(tabulate(res, res_headers))
    #np.savetxt(osp.join(globals.models_dir, "{}.csv".format(self.session_name)), np.asarray(res), delimiter=",", header=",".join(res_headers), comments='', fmt="%6.3f")
    np.savetxt(osp.join(globals.models_dir, "{}.csv".format(self.session_name)), np.asarray([res_headers] + res), delimiter=",", fmt="%s")

  # Train the model for an epoch
  def __train_epoch(self):
    self.model.train()

    time1 = time.time()
    losses = utils.LeveledAverageMeter(2)

    # Iterate over the dataset
    n_iter = len(self.dataloader)

    for i_iter,(net_input, gt_soP_prior, gt_pred_sem, loss_targets) in enumerate(self.dataloader):

      # Every now and then, print current loss avg
      if utils.smart_frequency_check(i_iter, n_iter, self.args.training.print_freq):
        print("\t{:4d}/{:<4d}: ".format(i_iter, n_iter), end="")

      # Ship tensors to GPU
      net_input    = net_input_to(net_input, utils.device)
      loss_targets = loss_targets_to(loss_targets, utils.device)
      gt_soP_prior = gt_soP_prior.to(utils.device)
      # TODO change this as_tensor
      gt_pred_sem  = torch.as_tensor(gt_pred_sem, dtype=torch.long, device = utils.device)

      # Note: batch_size is 1 for now
      batch_size = loss_targets[next(iter(loss_targets))].size()[0]

      # Forward pass & Backpropagation step
      self.optimizer.zero_grad()
      model_output = self.model(*net_input)

      # Unpack output
      _, rel_scores, pred_sem = model_output

      # Compute loss. If more loss functions are used, they are averaged
      loss, num_losses = 0, 0
      if "mlab" in self.args.training.loss:
        num_losses += 1
        # TODO: fix the weird-shaped mlab_target in datalayer and remove this view thingy
        if not "mlab_no_prior" in self.args.training.loss:
          # Note that maybe introducing no_predicate may be better:
          #  After all, there may not be a relationship between two objects...
          #  And this would avoid dirtying up the predictions?
          # TODO: change some constant to the gt_soP_prior before factoring it into the loss?
          gt_soP_prior = -0.5 * ( gt_soP_prior + (1.0 / self.dataset.n_pred))
          loss += self.criterions["mlab"]((gt_soP_prior + rel_scores).view(batch_size, -1), loss_targets["mlab"])
        else:
          loss += self.criterions["mlab"]((rel_scores).view(batch_size, -1), loss_targets["mlab"])
      if "bcel" in self.args.training.loss:
        num_losses += 1
        loss += self.criterions["bcel"]((rel_scores).view(batch_size, -1), loss_targets["1-hots"].view(batch_size, -1))
      if "mse" in self.args.training.loss:
        num_losses += 1
        # TODO use the weighted embeddings of gt_soP_prior ?
        loss += self.criterions["mse"](pred_sem, gt_pred_sem)

      loss /= num_losses

      # Optimization step
      loss.backward()
      self.optimizer.step()

      # Track loss values and print them every now and then
      losses.update(loss.mean().item())
      if utils.smart_frequency_check(i_iter, n_iter, self.args.training.print_freq, last=True):
        print("LOSS: {: 6.3f}\n".format(losses.avg(0)), end="")
        losses.reset(0)

    self.state["loss"] = losses.avg(1)
    time2 = time.time()

    print("TRAIN Loss: {: 6.3f} Time: {}".format(self.state["loss"], utils.time_diff_str(time1, time2)))

    """ Residual comments
      # the gt_soP_prior here is a subset of the 100*70*70 dimensional so_prior array, which contains the predicate prob distribution for all object pairs
      # the gt_soP_prior here contains the predicate probability distribution of only the object pairs in this annotation
      # Also, it might be helpful to keep in mind that this data layer currently works for a single annotation at a time - no batching is implemented!
    """

  # Test predicate prediction (PREDCLS)
  def test_pre(self):
    recalls, dtime = self.eval.test_pre(self.model, self.args.eval.rec, by_predicates = (False if not self.args.eval.by_predicates else self.dataset.n_pred))
    if self.args.eval.by_predicates:
      recall_by_pred = tuple([r for recall in recalls for r in recall[1]])
      recalls        = tuple([recall[0] for recall in recalls])
    print("PRED TEST:")
    print(self.__get_header_format_str(self.gt_headers(self.args.eval.test_pre)).format(*recalls))
    print("TEST Time: {}".format(utils.time_diff_str(dtime)))
    if self.args.eval.by_predicates: return recalls, recall_by_pred, dtime
    else:                            return recalls, dtime

  # Test relationship detection (RELDET)
  def test_rel(self):
    recalls, (pos_num, loc_num, gt_num), dtime = self.eval.test_rel(self.model, self.args.eval.rec, by_predicates = (False if not self.args.eval.by_predicates else self.dataset.n_pred))
    if self.args.eval.by_predicates:
      recall_by_pred = tuple([r for recall in recalls for r in recall[1]])
      recalls        = tuple([recall[0] for recall in recalls])
    print("REL TEST:")
    print(self.__get_header_format_str(self.gt_headers(self.args.eval.test_rel)).format(*recalls))
    if self.args.eval.eval_obj:
      print("OBJ TEST: POS: {: 6.3f}, LOC: {: 6.3f}, GT: {: 6.3f}, Precision: {: 6.3f}, Recall: {: 6.3f}".format(pos_num, loc_num, gt_num, np.float64(pos_num)/(pos_num+loc_num), np.float64(pos_num)/gt_num))
    print("TEST Time: {}".format(utils.time_diff_str(dtime)))
    if self.args.eval.by_predicates: return recalls, recall_by_pred, dtime
    else:                            return recalls, dtime

  # Helper functions
  def __get_header_format_str(self, gt_headers):
    return "".join(["\t{}: {{: 6.3f}}".format(x) if i % 2 == 0 else "\t{}: {{: 6.3f}}\n".format(x) for i,x in enumerate(gt_headers)])

  def gt_headers(self, test_type, prefix=""):
    def metric_name(x):
      if isinstance(x, float): return "{}x".format(x)
      elif isinstance(x, int): return str(x)
    if prefix == "":        fst_col_prefix = ""
    elif test_type != True: fst_col_prefix = "{} {} ".format(prefix, test_type)
    else:                   fst_col_prefix = "{} ".format(prefix)
    headers = []
    for i,x in enumerate(self.args.eval.rec):
      if i == 0: name = fst_col_prefix
      else:      name = ""
      name += "R@" + metric_name(x)
      headers.append(name)
      headers.append("{} ZS".format(name))
    return headers

# This class execute a VRDTrainer session for a number of times and averages the results
def VRDTrainerRepeater(repeat_n_times, **kwargs):

  # Obtain result headers & tables
  trainer = None
  results = []
  for v_iter in range(repeat_n_times):
    trainer = VRDTrainer(**kwargs)
    results.append(trainer.train(output_results = True))
  res_headerss, res_sheets = [i for i in zip(*results)]

  # Obtain any header
  res_headers = res_headerss[0]
  #for any_res_headers in res_headerss:
  #  assert np.array_equal(res_headers, any_res_headers), "Warning! Headers from repeated runs do not match: {}\n\n{},{}".format(res_headerss,res_headers,any_res_headers)

  # Compute average and deviation of the tables
  def get_avg_and_std(res_tables):
    def prepend_col(table, col):
      new_table = np.zeros((table.shape[0], table.shape[1]+1))
      new_table[:,1:] = table
      new_table[:,0] = col
      return new_table
    avg_table = res_tables.mean(axis=0)
    stds = res_tables[:,:,1:].std(axis=0)
    std_table = prepend_col(stds, avg_table[:,0])
    return avg_table, std_table

  output_xls = osp.join(globals.models_dir, "{}-r{}.xls".format(trainer.session_name, repeat_n_times))
  writer = pd.ExcelWriter(output_xls)
  writer_opt = {"float_format" : "%.2f"}

  res_sheets = utils.ld_to_dl(res_sheets)

  for table_name,res_tables in res_sheets.items():
    avg_table, std_table = get_avg_and_std(np.array(res_tables))
    avg = np.vstack((res_headers[table_name], avg_table))
    std = np.vstack((res_headers[table_name], std_table))
    if table_name in ["predicates", "predicates_stacked"]:
      avg, std = avg.transpose(), std.transpose()
    pd.DataFrame(avg).fillna("").to_excel(writer, sheet_name="{}-Avg".format(table_name), **writer_opt)
    pd.DataFrame(std).fillna("").to_excel(writer, sheet_name="{}-Dev".format(table_name), **writer_opt)

  writer.save()
  # TODO: add counts before the first epoch!
  # TODO: create, for each of the lines in avg_table (epoch) two sheets with as many columns as there are predicates, and stack the 4+1 1d arrays onto each other. Then, transpose.

if __name__ == "__main__":

  test_type = True # 0.2

  # DEBUGGING: Test if code compiles and a training session runs till the end
  if TEST_DEBUGGING:
    print("########################### TEST_DEBUGGING ###########################")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    trainer = VRDTrainer("test", {"training" : {"num_epochs" : 1, "test_first" : True}, # , "loss" : "mlab_bcel"},
        "eval" : {"test_pre" : test_type,  "test_rel" : test_type},
        #"model" : {"use_pred_sem" : 1+8},
        "data" : {"justafew" : True}}, profile="by_pred") #, checkpoint="epoch_4_checkpoint.pth.tar")
    trainer.train()


  # TEST_TRAIN_VALIDITY or TEST_EVAL_VALIDITY:
  #  Test if a newly-introduced change affects the validity of the evaluation (or evaluation+training)
  if TEST_TRAIN_VALIDITY or TEST_EVAL_VALIDITY:
    print("########################### TEST_VALIDITY ###########################")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dataset_name = "vrd" # /dsr"
    if TEST_EVAL_VALIDITY:
      VRDTrainer("original-checkpoint", {"training" : {"num_epochs" : 1, "test_first" : True}, "eval" : {"test_pre" : test_type,  "test_rel" : test_type},  "data" : {"name" : dataset_name}, "model" : {"feat_used" : {"spat" : False}}}, checkpoint="epoch_4_checkpoint.pth.tar").train()
    if TEST_TRAIN_VALIDITY:
      VRDTrainer("original", {"training" : {"num_epochs" : 5, "test_first" : True}, "eval" : {"test_pre" : test_type,  "test_rel" : False}, "data" : {"name" : dataset_name}}).train()


  torch.backends.cudnn.deterministic = False
  torch.backends.cudnn.benchmark = True
  random.seed(datetime.now())
  np.random.seed(int(time.time()))
  torch.manual_seed(int(time.time()))


  # TEST_OVERFIT: Try overfitting the network to a single batch
  if TEST_OVERFIT:
    print("########################### TEST_OVERFIT ###########################")
    justafew = 3
    args = {"training" : {"num_epochs" : 6}, "eval" : {"test_pre" : test_type,  "test_rel" : test_type, "justafew" : justafew},  "data" : {"justafew" : justafew}}
    VRDTrainer("test-overfit", args = args, profile = ["vg", "pred_sem"]).train()

    # Test VG
    #trainer = VRDTrainer("original-vg", {"training" : {"test_first" : True, "num_epochs" : 5}, "eval" : {"test_pre" : False, "test_rel" : test_type}}, profile = "vg")
    #trainer = VRDTrainer("original-vg", {"training" : {"test_first" : True, "num_epochs" : 5}, "eval" : {"test_pre" : test_type}}, profile = "vg")
    #trainer.train()


  ############################################################
  ## Scans
  ############################################################
  ## The following portion of code is useful for tuning the model
  ############################################################

  scan_name = "v19-all_preds-nored-test-repeater"
  v = 5
  base_profile = ["pred_sem", "by_pred"]
  base_training = {"num_epochs" : 5, "test_freq" : [2,3,4]}

  # Feature scan: scans different combinations of the features
  if FEATURES_SCAN:
    # VRDTrainer("test-no_prior-no-features",  {"training" : {"test_first" : True, "loss" : "mlab_no_prior"}}, profile = base_profile + ["no_feat"]).train()
    # VRDTrainer("test-no_prior-only_vis",  {"training" : {"num_epochs" : 4, "loss" : "mlab_no_prior"}, "model" : {"feat_used" : {"sem" : 0, "spat" : 0}}}).train()
    # VRDTrainer("test-no_prior-only_sem",  {"training" : {"num_epochs" : 4, "loss" : "mlab_no_prior"}, "model" : {"feat_used" : {"vis" : False, "vis_so" : False, "spat" : 0}}}).train()
    VRDTrainer("{}-test-no-features".format(scan_name),  {"training" : {"num_epochs" : 5, "test_freq" : [2,3,4], "test_first": True}}, profile = base_profile + ["no_feat"]).train()
    VRDTrainer("{}-test-only_spat".format(scan_name),    {"training" : base_training}, profile = base_profile + ["only_spat"]).train()

  # Parameters scan: scans parameter combinations
  if PARAMS_SCAN:
    print("PARAMS_SCAN")
    # Name of the embedding model in use
    for emb_model in ["gnews"]: # , "300", "glove-50" "50", "coco-70-50", "coco-30-50", "100"]:
      #if FEATURES_SCAN:
      #VRDTrainer("{}-test-only_sem-{}".format(scan_name, emb_model),  {"training" : base_training}, profile = base_profile + ["only_sem"]).train()

      # Default learning rate
      for lr in [0.0001]: # [0.001, 0.0001, 0.00001, 0.000001]:
        # Weight decay
        for weight_decay in [0.0001]:
          # multiplicative constant for the learning rate of the fusion layer (i.e second last layer)
          for lr_fus_ratio in [10]:
            # multiplicative constant for the learning rate of the scoring layer (i.e last layer)
            for lr_rel_ratio in [10]: #, 100]:
              # Predicate Semantics Mode, offset by one
              #  # -1 indicates no use of predicate semantics;
              #  # Values from 0 onwards indicate some of the different "modes" to introducte predicate semantics (e.g SemSim, Semantic Rescoring)
              for pred_sem_mode_1 in [-1, 16]: # , 3, 11]: #, 16+4, 16+2 , 16+4+1, 16+16+2, 16+16+4+2]: #, 9 16+16, 16+16+4
                # Loss function in use. These are the available ones
                #  # "mlab": MultiLabelMarginLoss
                #  # "mlab_no_prior": MultiLabelMarginLoss without soP_prior
                #  # "bcel": BCEWithLogitsLoss
                #  # "mse":  MSELoss
                # Loss functions can be used together by joining the two strings, for example with an underscore:
                #  # For instance, "mlab_mse" indicates using the average of MultiLabelMarginLoss and MSELoss as the loss
                for loss in ["mlab"]: # "bcel, mlab_mse]:
                  # Dataset in use. "vrd", "vg" # TODO check if "vrd:spatial" works
                  for dataset in ["vrd:spatial", "vrd:activities"]:
                    # Training profile to load. The profiles are listed in the ./cfgs/ folder, and they contain the options that are used to override the default ones (deafult.yml).
                    # Some examples are:
                    #  # "only_sem": Only uses semantic, "hides" visual and spatial features
                    #  # "only_spat": Only uses spatial features, "hides" visual and semantic features
                    #  # "sem_spat": Only uses semantic + spatial features, "hides" visual features
                    #  # "all_feats": Uses semantics + spatial + visual features
                    #  # "no_feat": Doesn't use features. Weird
                    for profile_name in ["only_sem"]: # "only_sem_subdot", "only_sem_catdiff", "only_sem_catdot", "only_sem_diffdot"]: # ["only_spat", "spat_sem", "only_sem", False]: # , "vg"]:
                      if "mse" in loss and (pred_sem_mode_1 == -1 or pred_sem_mode_1>=16):
                        continue

                      # Training session ID
                      session_id = "{}-{}-{}-{}-{}-{}-{}-{},{:b}-{}-{}".format(scan_name, emb_model, profile_name, lr, weight_decay, lr_fus_ratio, lr_rel_ratio, pred_sem_mode_1, pred_sem_mode_1, dataset, loss)
                      print("Session: ", session_id)

                      pred_sem_mode = pred_sem_mode_1+1
                      profile = base_profile + utils.listify(profile_name)
                      training = deepcopy(base_training)

                      #if dataset == "vg":
                      #  profile.append("vg")
                      #  training = {"num_epochs" : 4, "test_freq" : [1,2,3]}

                      # More to learn with all_feats?
                      if dataset == "vrd" and "all_feats" in profile and pred_sem_mode_1 >= 0 and pred_sem_mode_1 <= 16:
                        training["num_epochs"] += 1
                        training["test_freq"] = [x+1 for x in training["test_freq"]]
                      
                      training["loss"] = loss

                      # A training session takes:
                      #  # A session name, which will be used to label the saved results
                      #  # A dictionary specifying the options that override the profile
                      #  # A profile (string or list of strings) specifying the profile file(s) that are loaded and override(s) the default options (deafult.yml).
                      VRDTrainerRepeater(v, session_name = session_id, args = {
                          "data" : { "name" : dataset, "emb_model" : emb_model},
                          "training" : training,
                          "model" : {"use_pred_sem" : pred_sem_mode},
                          "eval" : {"test_pre" : True}, # "test_rel" : True},
                          #"eval" : {"test_pre" : False, "test_rel" : True},
                          "opt": {
                            "lr": lr,
                            "weight_decay" : weight_decay,
                            "lr_fus_ratio" : lr_fus_ratio,
                            "lr_rel_ratio" : lr_rel_ratio,
                            }
                          }, profile = profile)

  pass

  sys.exit(0)
