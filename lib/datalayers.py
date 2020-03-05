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

import utils
# TODO: expand so that it supports batch sizes > 1

class VRDDataLayer():
  """ Iterate through the dataset and yield the input and target for the network """

  def __init__(self, ds_info, stage):

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

    self.n_obj   = self.dataset.n_obj
    self.n_pred  = self.dataset.n_pred

    self.imgrels = [(k,v) for k,v in self.dataset.getImgRels().items()]
    self.n_imgrels = len(self.imgrels)
    self.cur_imgrels = 0

  #def __iter__(self):
  #    return self

  def __next__(self):

    (im_id, rels) = self.imgrels[self.cur_imgrels]

    im = utils.read_img(osp.join(self.dataset.img_dir, im_id))
    ih = im.shape[0]
    iw = im.shape[1]

    image_blob, im_scale = prep_im_for_blob(im, utils.vrd_pixel_means)
    img_blob = np.zeros((1,) + image_blob.shape, dtype=np.float32)
    img_blob[0] = image_blob

    # TODO: instead of computing boxes_img here, put it in the preprocess
    #  (and maybe transform relationships to contain object indices instead of whole objects)
    objs = []
    for i_rel,rel in enumerate(rels):
      objs.append(rel["subject"])
      objs.append(rel["object"])

    n_objs = len(objs)

    boxes_img = np.zeros(n_objs, )

    for i_obj,obj in enumerate(objs):
      boxes_img[i_obj] = obj["bbox"]

    # Objects' boxes
    so_boxes = np.zeros((n_objs, 5))
    so_boxes = np.zeros((boxes_img.shape[0], 5)) # , dtype=np.float32)
    so_boxes[:, 1:5] = boxes_img * im_scale


    # the dimension 8 here is the size of the spatial feature vector, containing the relative location and log-distance
    spatial_features = np.zeros((n_rel, 8))
    # the dimension 8 here is the size of the spatial feature vector, containing the relative location and log-distance
    semantic_features = np.zeros((n_rel, 2*300))

    # this will contain the probability distribution of each subject-object pair ID over all 70 predicates
    # rel_soP_prior = np.zeros((n_rel, self.dataset.n_pred))

    target = np.zeros((n_rel, self.n_pred))

    for i_rel,rel in enumerate(rels):

      rel["subject"]

      # these are the subject and object bounding boxes
      sBBox = utils.bboxDictToNumpy(rel["subject"]["bbox"])
      oBBox = utils.bboxDictToNumpy(rel["object"]["bbox"])

      # store the scaled dimensions of the union bounding box here, with the id i_rel
      spatial_features[i_rel] = utils.getRelativeLoc(sBBox, oBBox)

      # semantic features of obj and subj
      # semantic_features[i_rel] = utils.getSemanticVector(rel["subject"]["name"], rel["object"]["name"], self.w2v_model)
      semantic_features[i_rel] = np.zeros(600)

      # store the probability distribution of this subject-object pair from the soP_prior
      # if self.soP_prior != None:
      #   rel_soP_prior[i_rel] = self.dataset.soP_prior[classes[s_idx], classes[o_idx]]
      # else:
      #   rel_soP_prior[i_rel,:] = 0.0

      target[i_rel][rel["predicate"]["id"]] = 1.

      i_rel += 1

    self.cur_imgrels += 1
    if(self.cur_imgrels >= self.n_imgrels):
      self.cur_imgrels = 0


    # Note: the transpose should move the color channel to being the
    #  last dimension
    img_blob          = torch.FloatTensor(img_blob).permute(0, 3, 1, 2).cuda()
    so_boxes          = torch.FloatTensor(so_boxes).cuda()
    spatial_features  = torch.FloatTensor(spatial_features).cuda()
    semantic_features = torch.FloatTensor(semantic_features).cuda()
    target            = torch.LongTensor(target).cuda()


    yield img_blob
    yield so_boxes
    yield spatial_features
    yield semantic_features
    yield target

if __name__ == '__main__':
  pass
