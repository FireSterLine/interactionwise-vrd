import numpy as np
import torch

from lib.datalayers import VRDDataLayer
import lib.datalayers
from lib.evaluation_dsr import eval_recall_at_N, eval_obj_img # TODO remove this module
import time
import pickle
import os.path as osp
import torch.nn.functional as F
import warnings
import utils
from random import sample

# from copy import deepcopy
deepcopy = lambda x: x

# TODO: check, is all of this using the GPU, or can we improve the time?
class VRDEvaluator():
  """ Evaluator for Predicate Prediction and Relationship Prediction """

  def __init__(self, data_args, args):
    self.data_args = data_args
    self.args = args

    # Default args
    self.args.test_pre      = self.args.get("test_pre", True)
    self.args.test_rel      = self.args.get("test_rel", True)
    self.args.use_obj_prior = self.args.get("use_obj_prior", True)

    # Setup PREDICATE PREDICTION Data Layer
    if self.args.test_pre:
      self.datalayer_pre  = VRDDataLayer(data_args, "test", use_preload = self.args.use_preload)
      self.dataloader_pre = torch.utils.data.DataLoader(
        dataset = self.datalayer_pre,
        batch_size = 1, # 256,
        shuffle = False,
      )

    # Setup RELATIONSHIP DETECTION Data Layer
    if self.args.test_rel:
      self.datalayer_rel  = VRDDataLayer(data_args, "test", use_preload = self.args.use_preload, use_proposals = True)
      self.dataloader_rel = torch.utils.data.DataLoader(
        dataset = self.datalayer_rel,
        batch_size = 1, # 256,
        shuffle = False,
      )

    #self.datalayer  = VRDDataLayer(data_args, "test", use_preload = self.args.use_preload, use_proposals = self.args.test_rel)
    #self.dataloader = torch.utils.data.DataLoader(
    #  dataset = self.datalayer,
    #  batch_size = 1, # 256,
    #  shuffle = False,
    #)

    self.gt    = None
    self.gt_zs = None

    if self.args.test_pre or self.args.test_rel:
      self.any_datalayer = None
      if self.args.test_pre:
        self.any_datalayer = self.datalayer_pre
      elif self.args.test_rel:
        self.any_datalayer = self.datalayer_rel
    if True:
      try:
        # TODO: solve these by using the same dataset as the training one
        if self.any_datalayer.dataset.name == "vg":
          raise FileNotFoundError()
        # Load ground truths
        self.gt    = self.any_datalayer.dataset.readPKL(osp.join("eval", "gt.pkl"))
        self.gt_zs = self.any_datalayer.dataset.readPKL(osp.join("eval", "gt_zs.pkl"))
        #print(self.gt.keys())
        #print(type(self.gt))
        if self.args.justafew != False and isinstance(self.args.justafew, int):
          x =self.args.justafew
          print(self.gt["tuple_label"][self.args.justafew].shape)
          self.gt    = {'tuple_label' : [self.gt['tuple_label'][x]],    'obj_bboxes' : [self.gt['obj_bboxes'][x]],    'sub_bboxes' : [self.gt['sub_bboxes'][x]]}
          print(self.gt["tuple_label"][0].shape)
          self.gt_zs = {'tuple_label' : [self.gt_zs['tuple_label'][x]], 'obj_bboxes' : [self.gt_zs['obj_bboxes'][x]], 'sub_bboxes' : [self.gt_zs['sub_bboxes'][x]]}
      except FileNotFoundError:
        warnings.warn("Warning! Couldn't find ground truths pickles. Evaluation will be skipped.")

    # If None, the num_imgs that will be used is the size of the ground-truths
    self.num_imgs = None

    # VG is too slow, so we only test part of it
    if(self.data_args.name == "vg"):
      self.num_imgs = None
      # TODO: self.num_imgs = 8995

  def test_pre(self, vrd_model, Ns = [100, 50]):
    """ Test model on Predicate Prediction """
    if self.gt is None:
      return np.nan, np.nan, np.nan, np.nan, 0.1
    with torch.no_grad():
      vrd_model.eval()
      time1 = time.time()

      gt         = self.gt
      gt_zs      = self.gt_zs
      index = range(len(self.dataloader_pre))
      if isinstance(self.args.test_pre, float):
        index = sorted(sample(index, max(int(self.args.test_pre*len(index)),1)))
        gt    = {'tuple_label' : np.array(gt['tuple_label'])[index],    'obj_bboxes' : np.array(gt['obj_bboxes'])[index],    'sub_bboxes' : np.array(gt['sub_bboxes'])[index]}
        gt_zs = {'tuple_label' : np.array(gt_zs['tuple_label'])[index],    'obj_bboxes' : np.array(gt_zs['obj_bboxes'])[index],    'sub_bboxes' : np.array(gt_zs['sub_bboxes'])[index]}

      rlp_labels_cell  = []
      tuple_confs_cell = []
      sub_bboxes_cell  = []
      obj_bboxes_cell  = []

      N = 100 # What's this? (num of rel_res) (with this you can compute R@i for any i<=N)

      for (i_iter,(net_input, gt_obj, _, _)) in enumerate(dataloader):

        if i_iter not in index: continue
        if(isinstance(net_input, torch.Tensor) and net_input.size() == (1,)): # Check this one TODO
          rlp_labels_cell.append(None)
          tuple_confs_cell.append(None)
          sub_bboxes_cell.append(None)
          obj_bboxes_cell.append(None)
          continue

        if utils.smart_frequency_check(i_iter, len(dataloader), 0.1):
          print("{}/{}\r".format(i_iter, len(dataloader)), end="")

        (gt_obj_classes, gt_obj_boxes) = gt_obj
        net_input = lib.datalayers.net_input_to(net_input, utils.device)
        img_blob, obj_classes, obj_boxes, u_boxes, idx_s, idx_o, spatial_features = net_input

        tuple_confs_im = np.zeros((N,),   dtype = np.float) # Confidence...
        rlp_labels_im  = np.zeros((N, 3), dtype = np.float) # Rel triples
        sub_bboxes_im  = np.zeros((N, 4), dtype = np.float) # Subj bboxes
        obj_bboxes_im  = np.zeros((N, 4), dtype = np.float) # Obj bboxes

        _, rel_scores = vrd_model(*net_input)

        # TODO: fix when batch_size>1
        idx_s           = deepcopy(idx_s[0])
        idx_o           = deepcopy(idx_o[0])
        gt_obj_classes  = deepcopy(gt_obj_classes[0])
        gt_obj_boxes    = deepcopy(gt_obj_boxes[0])
        rel_scores      = deepcopy(rel_scores[0])
        #print(gt_obj_classes.shape)
        #print(gt_obj_boxes.shape)
        #print(obj_classes.shape)
        #print(obj_boxes.shape)
        #print(idx_s)
        #print(idx_o)
        rel_prob = rel_scores.data.cpu().numpy() # Is this the correct way?
        rel_res = np.dstack(np.unravel_index(np.argsort(-rel_prob.ravel()), rel_prob.shape))[0][:N]

        for ii in range(rel_res.shape[0]):
          rel = rel_res[ii, 1]
          tuple_idx = rel_res[ii, 0]

          conf = rel_prob[tuple_idx, rel]
          tuple_confs_im[ii] = conf

          rlp_labels_im[ii] = [gt_obj_classes[idx_s[tuple_idx]], rel, gt_obj_classes[idx_o[tuple_idx]]]

          sub_bboxes_im[ii] = gt_obj_boxes[idx_s[tuple_idx]]
          obj_bboxes_im[ii] = gt_obj_boxes[idx_o[tuple_idx]]

        tuple_confs_cell.append(tuple_confs_im)
        rlp_labels_cell.append(rlp_labels_im)
        sub_bboxes_cell.append(sub_bboxes_im)
        obj_bboxes_cell.append(obj_bboxes_im)

      res = {
        "rlp_confs_ours"  : tuple_confs_cell,
        "rlp_labels_ours" : rlp_labels_cell,
        "sub_bboxes_ours" : sub_bboxes_cell,
        "obj_bboxes_ours" : obj_bboxes_cell,
      }

      recalls = eval_recall_at_N(res, gts = [gt, gt_zs], Ns = Ns, num_imgs = self.num_imgs)
      time2 = time.time()

      return recalls, (time2-time1)

  # Relationship Prediction
  def test_rel(self, vrd_model, Ns = [100, 50]):
    """ Test model on Relationship Prediction """
    if self.gt is None:
      return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0.1
    with torch.no_grad():
      vrd_model.eval()
      time1 = time.time()

      N = 100 # What's this? (num of rel_res) (with this you can compute R@i for any i<=N)

      if self.args.eval_obj:
        pos_num = 0.0
        loc_num = 0.0
        gt_num  = 0.0

      gt         = self.gt
      gt_zs      = self.gt_zs
      index = range(len(self.dataloader_rel))
      if isinstance(self.args.test_rel, float):
        index = sorted(sample(index, max(int(self.args.test_rel*len(index)),1)))
        gt    = {'tuple_label' : np.array(gt['tuple_label'])[index],    'obj_bboxes' : np.array(gt['obj_bboxes'])[index],    'sub_bboxes' : np.array(gt['sub_bboxes'])[index]}
        gt_zs = {'tuple_label' : np.array(gt_zs['tuple_label'])[index],    'obj_bboxes' : np.array(gt_zs['obj_bboxes'])[index],    'sub_bboxes' : np.array(gt_zs['sub_bboxes'])[index]}

      rlp_labels_cell  = []
      tuple_confs_cell = []
      sub_bboxes_cell  = []
      obj_bboxes_cell  = []

      n_iter = len(self.dataloader_rel)
      for i_iter,test_data in enumerate(self.dataloader_rel):
        if i_iter not in index: continue

        net_input, gt_obj, det_obj, gt_soP_prior  = test_data
        if(isinstance(net_input, torch.Tensor) and net_input.size() == (1,)): # Check this one TODO
          rlp_labels_cell.append(None)
          tuple_confs_cell.append(None)
          sub_bboxes_cell.append(None)
          obj_bboxes_cell.append(None)
          continue

        if utils.smart_frequency_check(i_iter, n_iter, 0.1):
            print("{}/{}\r".format(i_iter,n_iter), end="")

        (gt_obj_classes, gt_obj_boxes) = gt_obj
        (det_obj_classes, det_obj_boxes, det_obj_confs) = det_obj
        # TODO: remove this to allow batching
        gt_obj_boxes  = gt_obj_boxes[0].data.cpu().numpy().astype(np.float32)
        gt_obj_classes = gt_obj_classes[0].data.cpu().numpy().astype(np.float32)

        net_input = lib.datalayers.net_input_to(net_input, utils.device)
        obj_score, rel_score = vrd_model(*net_input) # img_blob, obj_boxes, u_boxes, idx_s, idx_o, spatial_features, obj_classes)

        # TODO: remove and fix everyhing else to allow batching
        # obj_score = obj_score[0]
        #rel_score = rel_score[0]

        _, obj_pred = obj_score[:, 1::].data.topk(1, 1, True, True)

        # TODO: use this (what for? Just print, maybe) (it's dim=2 maybe?)
        obj_score = F.softmax(obj_score, dim=1)[:, 1::].data.cpu().numpy()

        rel_prob = rel_score.data.cpu().numpy()
        gt_soP_prior = gt_soP_prior.data.cpu().numpy()
        rel_prob += np.log(0.5*(gt_soP_prior+1.0 / self.any_datalayer.n_pred))


        img_blob, obj_classes, obj_boxes, u_boxes, idx_s, idx_o, spatial_features = net_input

        # print("gt_obj_boxes.shape: \t", gt_obj_boxes.shape)
        # print("gt_cls.shape: \t", gt_cls.shape)
        # print("det_obj_boxes.shape: \t", det_obj_boxes.shape)
        # print("obj_pred.shape: \t", obj_pred.shape)
        # print()
        # TODO: remove this to allow batching
        idx_s = deepcopy(idx_s[0])
        idx_o = deepcopy(idx_o[0])
        det_obj_boxes = deepcopy(det_obj_boxes[0])

        if self.args.eval_obj:
          pos_num_img, loc_num_img = eval_obj_img(gt_obj_boxes, gt_obj_classes, det_obj_boxes, obj_pred.cpu().numpy(), gt_thr=0.5)
          pos_num += pos_num_img
          loc_num += loc_num_img
          gt_num  += gt_obj_boxes.shape[0]

        # TODO: remove this to allow batching
        det_obj_confs   = deepcopy(det_obj_confs[0])
        rel_prob = deepcopy(rel_prob[0])
        det_obj_classes = deepcopy(det_obj_classes[0])

        tuple_confs_im = []
        rlp_labels_im  = np.zeros((rel_prob.shape[0]*rel_prob.shape[1], 3), dtype = np.float)
        sub_bboxes_im  = np.zeros((rel_prob.shape[0]*rel_prob.shape[1], 4), dtype = np.float)
        obj_bboxes_im  = np.zeros((rel_prob.shape[0]*rel_prob.shape[1], 4), dtype = np.float)
        n_idx = 0

        # print("det_res['confs'].shape: ", det_obj_confs.shape)
        # print("rel_prob.shape: ", rel_prob.shape)

        for tuple_idx in range(rel_prob.shape[0]):
          for rel in range(rel_prob.shape[1]):
            # print((tuple_idx, rel))
            # print("np.log(det_res['confs']).shape: ", np.log(det_obj_confs).shape)
            # print("np.log(det_res['confs'][idx_s[tuple_idx]]).shape: ", np.log(det_obj_confs[idx_s[tuple_idx]]).shape)
            # print("det_res['confs'].shape: ", det_obj_confs.shape)
            # print("det_res['confs'][idx_s[tuple_idx]].shape: ", det_obj_confs[idx_s[tuple_idx]].shape)
            # print("rel_prob[tuple_idx].shape: ", rel_prob[tuple_idx].shape)
            # print("rel_prob[tuple_idx, rel].shape: ", rel_prob[tuple_idx, rel].shape)
            if(self.args.use_obj_prior):
              if(det_obj_confs.ndim == 1):
                # Maybe we never reach this point? Or maybe it accounts for batching?
                conf = np.log(det_obj_confs[idx_s[tuple_idx]]) + np.log(det_obj_confs[idx_o[tuple_idx]]) + rel_prob[tuple_idx, rel]
              else:
                conf = np.log(det_obj_confs[idx_s[tuple_idx], 0]) + np.log(det_obj_confs[idx_o[tuple_idx], 0]) + rel_prob[tuple_idx, rel]
            else:
              conf = rel_prob[tuple_idx, rel]
            tuple_confs_im.append(conf)
            sub_bboxes_im[n_idx] = det_obj_boxes[idx_s[tuple_idx]]
            obj_bboxes_im[n_idx] = det_obj_boxes[idx_o[tuple_idx]]
            rlp_labels_im[n_idx] = [det_obj_classes[idx_s[tuple_idx]], rel, det_obj_classes[idx_o[tuple_idx]]]
            n_idx += 1

        # Why is this needed? ...
        tuple_confs_im = np.array(tuple_confs_im)
        idx_order = tuple_confs_im.argsort()[::-1][:N] # TODO: remove this :N to allow R@4x
        rlp_labels_im = rlp_labels_im[idx_order,:]
        tuple_confs_im = tuple_confs_im[idx_order]
        sub_bboxes_im  = sub_bboxes_im[idx_order,:]
        obj_bboxes_im  = obj_bboxes_im[idx_order,:]

        rlp_labels_cell.append(rlp_labels_im)
        tuple_confs_cell.append(tuple_confs_im)
        sub_bboxes_cell.append(sub_bboxes_im)
        obj_bboxes_cell.append(obj_bboxes_im)

      res = {
        "rlp_confs_ours"  : tuple_confs_cell,
        "rlp_labels_ours" : rlp_labels_cell,
        "sub_bboxes_ours" : sub_bboxes_cell,
        "obj_bboxes_ours" : obj_bboxes_cell,
      }

      # if len(len(self.dataloader)) != len(res["obj_bboxes_ours"]):
      #   warnings.warn("Warning! Rel test results and gt do not have the same length: rel test performance might be off! {} != {}".format(len(len(self.dataloader)), len(res["obj_bboxes_ours"])), UserWarning)

      recalls = eval_recall_at_N(res, gts = [gt, gt_zs], Ns = Ns, num_imgs = self.num_imgs)
      time2 = time.time()

      if self.args.eval_obj:
        return recalls, (pos_num, loc_num, gt_num), (time2 - time1)
      else:
        return recalls, (np.nan, np.nan, np.nan), (time2 - time1)
