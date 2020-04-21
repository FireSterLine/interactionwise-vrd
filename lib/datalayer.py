import numpy as np
import os.path as osp
import scipy.io as sio
import scipy

import pickle
from lib.blob import prep_im_for_blob
from lib.dataset import VRDDataset
import torch

import utils, globals
from copy import copy, deepcopy
from torch.utils import data

class VRDDataLayer(data.Dataset):
  """ Iterate through the dataset and yield the input and target for the network """

  def __init__(self, dataset, stage, use_proposals = False, use_preload = False, x_cols = [], y_cols = []):
    super(VRDDataLayer, self).__init__()

    self.dataset = dataset
    self.stage   = stage

    # TODO: Receive this parameter as an argument, don't hardcode this
    # self.granularity = "rel"
    self.granularity = "img"


    self.x_cols = x_cols
    self.y_cols = y_cols

    # TODO: remove these two?
    self.n_obj   = self.dataset.n_obj
    self.n_pred  = self.dataset.n_pred

    self.soP_prior = self.dataset.getDistribution("soP")

    # TODO: allow to choose which model to use. We only have w2v for now
    self.emb = {"obj" : self.dataset.readJSON("objects-emb.json"), "pred" : self.dataset.readJSON("predicates-emb.json")}

    self.vrd_data = self.dataset.getData("annos", self.stage, self.granularity)

    # TODO: change this when you switch to annos?
    def numrels(vrd_sample):
      if self.granularity == "rel":
        return 1
      elif self.granularity == "img":
        if isinstance(vrd_sample, list):
          return len(vrd_sample)
        elif isinstance(vrd_sample, dict):
          return len(vrd_sample["rels"]) > 0

    # TODO: check I should include images with no relationships
    # Ignore None elements during training
    if self.stage == "train":
      self.vrd_data = [(k,v) for k,v in self.vrd_data if k != None and numrels(v) > 0]


    self.N = len(self.vrd_data)
    self.wrap_around = ( self.stage == "train" )

    self.objdet_res = None
    if use_proposals != False:
      # TODO: proposals is not ordered, but a dictionary with im_path keys
      # TODO: expand so that we don't need the proposals pickle, and we generate it if it's not there, using Faster-RCNN?
      # TODO: move the proposals file path to a different one (maybe in Faster-RCNN)
      proposals = self.dataset.readPKL(osp.join("eval", "det_res.pkl"))
      # TODO fix data bottlenecks like this one
      self.objdet_res = [ {
          "boxes"   : proposals["boxes"][i],
          "classes" : proposals["cls"][i].reshape(-1),
          "confs"   : proposals["confs"][i].reshape(-1)
        } for i in range(len(proposals["boxes"]))]

    self.preloaded = None
    if use_preload:
      print("Preloading...")
      self.preloaded = [self.__computeitem__(index) for index in range(self.__len__())]

    # NOTE: the max shape is across all the dataset, whereas we would like for it to be for a unique batch.
    # TODO: write collate_fn using this code: https://pytorch.org/docs/stable/data.html#loading-batched-and-non-batched-data
    '''
    this is the max shape that was identified by running the function above on the
    VRD dataset. It takes too long to run, so it's better if we run this once on
    any new dataset, and store and initialize those values as done here
    '''
    # self.max_shape = (1000, 1000, 3)
    # self.max_shape = self._get_max_shape()
    # print("Max shape is: {}".format(self.max_shape))


  def _get_max_shape(self):
    print("Identifying max shape...")
    im_shapes = []
    for img_path, _ in self.vrd_data:
      if img_path is not None:
        im = self.dataset.readImg(img_path)
        image_blob, _ = prep_im_for_blob(im, utils.vrd_pixel_means)
        im_shapes.append(image_blob.shape)
    max_shape = np.array(im_shapes).max(axis=0)
    return max_shape

  def __len__(self):
    return self.N

  def __getitem__(self, index):
    # print("index: ", index)
    # Read/compute output
    if self.preloaded is None:
      return self.__computeitem__(index)
    else:
      return self.preloaded[index]

  def __computeitem__(self, index):

    # Helper for returning none
    def _None():
      if self.stage == "test":
        if self.objdet_res is None:
          return False, False, False, False
        else:
          return False, False, False, False
      elif self.stage == "train":
        print("Warning! I'm about to return None values during training. That's not good, batching will probably fail")
        return False, False, False, False

    (img_path, annos) = self.vrd_data[index]

    # TODO: probably False values won't allow batching. But this is not a problem in training because None and len(rels)==0 are ignored
    if img_path is None:
      return _None()


    # TODO: when granularity is rel, then it might be better to receive the thing in form of a relst and to create objs manually
    # if self.granularity == "rel":
    #   rels = [rels]

    objs     = annos["objs"]
    rels     = annos["rels"]

    n_objs, n_rels = len(objs), len(rels)

    # TODO: Wait a second, shouldn't we test anyway on an image with no
    #  relationships? I mean, if we identified some objects, why not? What does DSR do?
    if n_rels == 0:
      return _None()

    ###########################################################################
    ###########################################################################

    # Encode the boxes in a format that ROI Pooling layers accept
    #  See https://pytorch.org/docs/stable/torchvision/ops.html#torchvision.ops.roi_pool
    def bboxesToROIBoxes(boxes):
      roi_boxes = np.zeros((boxes.shape[0], 5))
      # Note: The ROI Pooling layer expects the index of the batch as first column
      # TODO Right now, this won't actually work for index>1 when shuffling...
      #  if you allow batching, then you must post-process the indices. Maybe in collate_fn!! That's the solution, yeah!
      roi_boxes[:, 0]   = index
      roi_boxes[:, 1:5] = boxes * im_scale
      return roi_boxes

    im = self.dataset.readImg(img_path)
    ih, iw = im.shape[0], im.shape[1]

    # TODO: enable this to temporarily allow batching (write collate_fn then)
    # image_blob, im_scale = prep_im_for_blob(im, utils.vrd_pixel_means)
    # blob = np.zeros(self.max_shape)
    # blob[0: image_blob.shape[0], 0:image_blob.shape[1], :] = image_blob
    # image_blob = blob
    # img_blob = np.zeros((1,) + image_blob.shape, dtype=np.float32)
    # img_blob[0] = image_blob

    # TODO: check if this works (it removes the first (not moot) dimension)
    image_blob, im_scale = prep_im_for_blob(im, utils.vrd_pixel_means)
    img_blob = image_blob
    # img_blob = np.zeros(self.max_shape, dtype=np.float32)
    # img_blob[: image_blob.shape[0], :image_blob.shape[1], :] = image_blob

    # OBJECTS

    # Ground-truths objects
    gt_obj_classes = np.zeros((n_objs))
    gt_obj_boxes   = np.zeros((n_objs, 4))

    for i_obj, obj in enumerate(objs):
      gt_obj_classes[i_obj] = obj["cls"]
      gt_obj_boxes[i_obj]  = utils.bboxListToNumpy(obj["bbox"])

    # object detections (if any)
    if self.objdet_res is not None:
      det_res = self.objdet_res[index]

      det_obj_classes  = det_res["classes"]
      det_obj_boxes    = det_res["boxes"]
      det_obj_confs    = det_res["confs"]

      n_rels = len(det_obj_classes) * (len(det_obj_classes) - 1)

      if n_rels == 0:
        return _None()


    # Union bounding boxes
    u_boxes = np.zeros((n_rels, 4))

    # Semantic features
    obj_spat_feat = np.zeros(1) # Dummy
    if "dsr_spat_vec" in self.x_cols:
      # Spatial vector containing the relative location and log-distance
      obj_spat_feat = np.zeros((n_rels, 8))
    elif "dsr_spat_mat" in self.x_cols:
      # Binary matrix displaying the relative location
      obj_spat_feat = np.zeros((n_rels, 2, 32, 32))

    # Prior distribution of the object pairs across all predicates
    gt_soP_prior = np.zeros((n_rels, self.dataset.n_pred))

    # Indices for objects and subjects
    idx_s, idx_o = [], []

    def addRel(i_rel,               \
                sub_idx,  obj_idx,  \
                sub_cls,  obj_cls,  \
                sub_bbox, obj_bbx):
      # Subject and object local indices (useful when selecting ROI results)
      idx_s.append(sub_idx)
      idx_o.append(obj_idx)

      # Subject and object bounding boxes
      sBBox = sub_bbox
      oBBox = obj_bbx

      # get the union bounding box
      rBBox = utils.getUnionBBox(sBBox, oBBox, ih, iw)

      # store the union box (= relation box) of the union bounding box here, with the id i_rel_inst
      u_boxes[i_rel] = np.array(rBBox) * im_scale

      # store the scaled dimensions of the union bounding box here, with the id i_rel
      if "dsr_spat_vec" in self.x_cols:
        obj_spat_feat[i_rel] = utils.getRelativeLoc(sBBox, oBBox)
      elif "dsr_spat_mat" in self.x_cols:
        obj_spat_feat[i_rel] = [utils.getDualMask(ih, iw, sBBox),
                                utils.getDualMask(ih, iw, oBBox)]

      # store the probability distribution of this subject-object pair from the soP_prior
      gt_soP_prior[i_rel] = self.soP_prior[sub_cls][obj_cls]

    if self.objdet_res is not None:
      i_rel = 0
      for sub_idx, sub_cls in enumerate(det_obj_classes):
        for obj_idx, obj_cls in enumerate(det_obj_classes):
          if (sub_idx == obj_idx): continue
          addRel(i_rel,                   \
                  sub_idx, obj_idx,       \
                  sub_cls, obj_cls,       \
                  det_obj_boxes[sub_idx], \
                  det_obj_boxes[obj_idx])
          i_rel += 1
    else:

      # Semantic vector for the predicate
      gt_pred_sem = np.zeros((n_rels, globals.emb_size))

      # Targets (ground-truth input to the losses)
      if "mlab" in self.y_cols:
        # TODO: reshape like mlab_target = np.zeros((n_rels, self.n_pred))
        mlab_target = -1 * np.ones((self.dataset.n_pred * n_rels))
        pos_idx = 0

      if "1-hots" in self.y_cols:
        onehots_target = torch.zeros(n_rels, self.dataset.n_pred)

      for i_rel, rel in enumerate(rels):
        addRel(i_rel,
             rel["sub"], rel["obj"],
             objs[rel["sub"]]["cls"],
             objs[rel["obj"]]["cls"],
             utils.bboxListToNumpy(objs[rel["sub"]]["bbox"]),
             utils.bboxListToNumpy(objs[rel["obj"]]["bbox"]))

        pred_clss = rel["pred"]

        # GROUND-TRUTHS

        # TODO: this is no good way of accounting for multi-label.
        gt_pred_sem[i_rel] = np.mean([self.emb["pred"][pred_cls] for pred_cls in pred_clss], axis=0)

        if "mlab" in self.y_cols:
          for pred_cls in pred_clss:
            mlab_target[pos_idx] = i_rel * self.dataset.n_pred + pred_cls
            pos_idx += 1
        if "1-hots" in self.y_cols:
          onehots_target[i_rel].scatter_(0, torch.tensor(pred_clss), 1.)

    if self.objdet_res is not None:
      obj_classes  = det_obj_classes
      obj_boxes    = det_obj_boxes
    else:
      obj_classes  = gt_obj_classes
      obj_boxes    = gt_obj_boxes

    roi_obj_boxes = bboxesToROIBoxes(obj_boxes)
    roi_u_boxes   = bboxesToROIBoxes(u_boxes)


    # img_blob          = torch.as_tensor(img_blob,        dtype=torch.float,    device = utils.device).permute(2, 0, 1)
    # img_blob   = np.transpose(img_blob,(2,0,1))

    img_blob          = torch.as_tensor(img_blob,        dtype=torch.float).permute(2, 0, 1)
    roi_obj_boxes     = torch.as_tensor(roi_obj_boxes,   dtype=torch.float)
    roi_u_boxes       = torch.as_tensor(roi_u_boxes,     dtype=torch.float)
    idx_s             = torch.as_tensor(idx_s,           dtype=torch.long)
    idx_o             = torch.as_tensor(idx_o,           dtype=torch.long)
    obj_spat_feat     = torch.as_tensor(obj_spat_feat,   dtype=torch.float)
    obj_classes       = torch.as_tensor(obj_classes,     dtype=torch.long)

    # gt_soP_prior      = torch.as_tensor(gt_soP_prior,    dtype=torch.float)


    net_input = (img_blob,
                 obj_classes,
                 roi_obj_boxes,
                 roi_u_boxes,
                 idx_s,
                 idx_o,
                 obj_spat_feat
      )

    gt_obj  = (gt_obj_classes,  gt_obj_boxes)

    if self.stage == "test":
      if self.objdet_res is None:
        return net_input, gt_obj, False, False
      else:
        det_obj = (det_obj_classes, det_obj_boxes, det_obj_confs)
        return net_input, gt_obj, det_obj, gt_soP_prior

    elif self.stage == "train":
      gt_pred_sem      = torch.as_tensor(gt_pred_sem,    dtype=torch.long)
      loss_targets = {}
      if "mlab" in self.y_cols:
        loss_targets["mlab"]   = torch.as_tensor(mlab_target,    dtype=torch.long)
      if "1-hots" in self.y_cols:
        loss_targets["1-hots"] = torch.as_tensor(onehots_target, dtype=torch.float)
      return net_input,       \
              gt_soP_prior,   \
              gt_pred_sem,    \
              loss_targets



# Move the net input to device
def loss_targets_to(loss_targets, device):
  loss_targets = deepcopy(loss_targets)
  for name,loss_target in loss_targets.items():
    loss_targets[name] = loss_target.to(device = device)
  return loss_targets

# Move the net input to device
def net_input_to(net_input, device):
  # Move to GPU
  if net_input is not False: # not (isinstance(net_input, torch.Tensor) and net_input.size() == (1,)):
    (img_blob,
             obj_classes,
             roi_obj_boxes,
             roi_u_boxes,
             idx_s,
             idx_o,
             obj_spat_feat,
    ) = net_input

    img_blob          = img_blob.to(device) # torch.as_tensor(img_blob,        dtype=torch.float,    device = device)
    obj_classes       = obj_classes.to(device) # torch.as_tensor(obj_classes,     dtype=torch.long,     device = device)
    roi_obj_boxes     = roi_obj_boxes.to(device) # torch.as_tensor(roi_obj_boxes,   dtype=torch.float,    device = device)
    roi_u_boxes       = roi_u_boxes.to(device) # torch.as_tensor(roi_u_boxes,     dtype=torch.float,    device = device)
    idx_s             = idx_s.to(device) # torch.as_tensor(idx_s,           dtype=torch.long,     device = device)
    idx_o             = idx_o.to(device) # torch.as_tensor(idx_o,           dtype=torch.long,     device = device)
    obj_spat_feat     = obj_spat_feat.to(device) # torch.as_tensor(obj_spat_feat,    dtype=torch.float,    device = device)

    net_input = (img_blob,
             obj_classes,
             roi_obj_boxes,
             roi_u_boxes,
             idx_s,
             idx_o,
             obj_spat_feat,
    )
  return net_input

"""
# Batching example:
data_info = {"name": "vrd", "with_bg_obj" : False, "with_bg_pred" : False}
datalayer = VRDDataLayer(data_info, "train")
train_generator = data.DataLoader(
  dataset = datalayer,
  batch_size = 2, # 256,
  shuffle = True
)

for i,a in enumerate(datalayer):
  if i > 1: break
  print(i, len(a))
  print([x.shape for x in a[0]], a[1].shape, a[2].shape)
  print()
  input()

for i, a in enumerate(train_generator):
  print(i, len(a))
  print([x.shape for x in a[0]], a[1].shape, a[2].shape)
  print()
  input()
  if i > 0: break
"""


"""
data_info = {"name": "vrd", "with_bg_obj": False, "with_bg_pred": False}
datalayer = VRDDataLayer(data_info, "train")
a = datalayer[0]
​
train_generator = data.DataLoader(
    dataset=datalayer,
    drop_last=True,
    batch_size=1,
    shuffle=True
)
"""

"""
TODO: make this an data.IterableDataset and allow parallelization?

  def __init__(self, data_info, stage):
    super(VRDDataLayer).__init__()
    ...
  ...
  def __iter__(self):
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is None:  # single-process data loading, return the full iterator
      iter_start = self.start
      iter_end = self.end
    else:  # in a worker process
      # split workload
      per_worker = int(math.ceil((self.end - self.start) / float(worker_info.num_workers)))
      worker_id = worker_info.id
      iter_start = self.start + worker_id * per_worker
      iter_end = min(iter_start + per_worker, self.end)
    return iter(range(iter_start, iter_end))

ds = VRDDataLayer(start=3, end=7)

print(list(torch.utils.data.DataLoader(ds, num_workers=self.num_workers... but note that this messes up the order ( see https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset ))))

"""
