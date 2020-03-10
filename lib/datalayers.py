import numpy as np
import os.path as osp
import scipy.io as sio
import scipy
import cv2
import pickle
import sys
from lib.blob import prep_im_for_blob
from lib.dataset import dataset
import math
import torch
from gensim.models import KeyedVectors

import utils, globals
from copy import copy, deepcopy
# TODO: expand so that it supports batch sizes > 1

class VRDDataLayer():
  """ Iterate through the dataset and yield the input and target for the network """

  def __init__(self, ds_info, stage):
    ds_info = copy(ds_info)
    if isinstance(ds_info, str):
      ds_name = ds_info
      ds_args = {}
    else:
      ds_name = ds_info["ds_name"]
      del ds_info["ds_name"]
      ds_args = ds_info

    self.ds_name = ds_name
    self.stage   = stage

    self.dataset = dataset(self.ds_name, **ds_args)
    self.soP_prior = self.dataset.getDistribution(type="soP", force=True)

    self.n_obj   = self.dataset.n_obj
    self.n_pred  = self.dataset.n_pred

    self.imgrels   = deepcopy([(k,v) for k,v in self.dataset.getImgRels(self.stage).items()])
    # self.imgrels   = deepcopy([(k,v) for k,v in self.dataset.getImgRels(self.stage).items()])[:10]
    self.n_imgrels = len(self.imgrels)
    self.cur_imgrels = 0
    self.wrap_around = ( self.stage == "train" )

    self.batch_size = 1
    # TODO: take care of the remaining
    self.n_imgrel_batches = self.n_imgrels // self.batch_size

    # print("Loading Word2Vec model...")
    # self.w2v_model = KeyedVectors.load_word2vec_format(osp.join(globals.data_dir, globals.w2v_model_path), binary=True)

  def __iter__(self):
      return self

  def __next__(self):

    if self.cur_imgrels >= self.n_imgrels:
      if self.wrap_around:
        self.cur_imgrels = 0
      else:
        raise StopIteration
        return

    (im_id, _rels) = self.imgrels[self.cur_imgrels]

    rels = deepcopy(_rels)

    im = utils.read_img(osp.join(self.dataset.img_dir, im_id))
    ih = im.shape[0]
    iw = im.shape[1]

    image_blob, im_scale = prep_im_for_blob(im, utils.vrd_pixel_means)
    img_blob = np.zeros((1,) + image_blob.shape, dtype=np.float32)
    img_blob[0] = image_blob


    # TODO: instead of computing boxes_img here, put it in the preprocess
    #  (and maybe transform relationships to contain object indices instead of whole objects)
    # Note: from here on, rel["subject"] and rel["object"] contain indices to objs
    objs = []
    boxes = []
    for i_rel,rel in enumerate(rels):

      i_obj = len(objs)
      objs.append(rel["subject"])
      rel["subject"] = i_obj

      i_obj = len(objs)
      objs.append(rel["object"])
      rel["object"] = i_obj

    n_objs = len(objs)

    boxes_img = np.zeros((n_objs, 4))
    classes_img = np.zeros((n_objs))

    for i_obj,obj in enumerate(objs):
      boxes_img[i_obj] = utils.bboxDictToNumpy(obj["bbox"])
      classes_img[i_obj] = obj["id"]



    # Objects' boxes
    obj_boxes = np.zeros((boxes_img.shape[0], 5)) # , dtype=np.float32)
    obj_boxes[:, 1:5] = boxes_img * im_scale


    n_rel = len(rels)

    # union bounding boxes
    u_boxes = np.zeros((n_rel, 5))

    # the dimension 8 here is the size of the spatial feature vector, containing the relative location and log-distance
    spatial_features = np.zeros((n_rel, 8))
    # TODO: introduce the other spatial feature thingy
    # spatial_features = np.zeros((n_rel, 2, 32, 32))
    #     spatial_features[ii] = [self._getDualMask(ih, iw, sBBox), \
    #               self._getDualMask(ih, iw, oBBox)]

    # TODO: add tiny comment...
    semantic_features = np.zeros((n_rel, 2*300))

    # this will contain the probability distribution of each subject-object pair ID over all 70 predicates
    rel_soP_prior = np.zeros((n_rel, self.dataset.n_pred))

    # Target output for the network
    target = -1*np.ones((1, self.dataset.n_pred*n_rel))
    pos_idx = 0
    # target = np.zeros((n_rel, self.n_pred))

    # Indices for objects and subjects
    idx_s,idx_o = [],[]

    for i_rel,rel in enumerate(rels):

      # Subject and object bounding boxes
      sBBox = utils.bboxDictToNumpy(objs[rel["subject"]]["bbox"])
      oBBox = utils.bboxDictToNumpy(objs[rel["object"]]["bbox"])

      # get the union bounding box
      rBBox = utils.getUnionBBox(sBBox, oBBox, ih, iw)
      # store the union box (= relation box) of the union bounding box here, with the id i_rel_inst
      u_boxes[i_rel, 1:5] = np.array(rBBox) * im_scale

      # Subject and object local indices (useful when selecting ROI results)
      idx_s.append(rel["subject"])
      idx_o.append(rel["object"])

      # store the scaled dimensions of the union bounding box here, with the id i_rel
      spatial_features[i_rel] = utils.getRelativeLoc(sBBox, oBBox)

      # semantic features of obj and subj
      # semantic_features[i_rel] = utils.getSemanticVector(objs[rel["subject"]]["name"], objs[rel["object"]]["name"], self.w2v_model)
      semantic_features[i_rel] = np.zeros(600)

      # store the probability distribution of this subject-object pair from the soP_prior
      s_cls_id = objs[rel["subject"]]["id"]
      o_cls_id = objs[rel["object"]]["id"]
      rel_soP_prior[i_rel] = self.soP_prior[s_cls_id][o_cls_id]

      # TODO: this target is not the one we want
      # target[i_rel][rel["predicate"]["id"]] = 1.
      # TODO: enable multi-class predicate (rel_classes: list of predicates for every pair)
      rel_classes = [rel["predicate"]["id"]]
      for rel_label in rel_classes:
        target[0, pos_idx] = i_rel*self.dataset.n_pred + rel_label
        pos_idx += 1

      i_rel += 1

    self.cur_imgrels += 1


    # print(target)

    # Note: the transpose should move the color channel to being the
    #  last dimension
    img_blob          = torch.FloatTensor(img_blob).permute(0, 3, 1, 2).cuda()
    obj_boxes         = torch.FloatTensor(obj_boxes).cuda()
    u_boxes           = torch.FloatTensor(u_boxes).cuda()
    idx_s             = torch.LongTensor(idx_s).cuda()
    idx_o             = torch.LongTensor(idx_o).cuda()
    spatial_features  = torch.FloatTensor(spatial_features).cuda()
    semantic_features = torch.FloatTensor(semantic_features).cuda()

    rel_soP_prior = torch.FloatTensor(rel_soP_prior).cuda()
    target        = torch.LongTensor(target).cuda()
    # target        = torch.LongTensor(target).cuda()


    yield img_blob
    yield obj_boxes
    yield u_boxes
    yield idx_s
    yield idx_o
    yield spatial_features
    yield semantic_features

    if self.stage == "train":
      yield rel_soP_prior
      yield target
    elif self.stage == "test":
      yield classes_img
      yield boxes_img

if __name__ == '__main__':
  pass
