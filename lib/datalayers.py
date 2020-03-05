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

    image_blob, im_scale = prep_im_for_blob(im, globals.vrd_pixel_means)
    img_blob = np.zeros((1,) + image_blob.shape, dtype=np.float32)
    img_blob[0] = image_blob

    n_rel = len(rels)

    # the dimension 8 here is the size of the spatial feature vector, containing the relative location and log-distance
    spatial_features = np.zeros((n_rel, 8))
    # the dimension 8 here is the size of the spatial feature vector, containing the relative location and log-distance
    semantic_features = np.zeros((n_rel, 2*300))

    # this will contain the probability distribution of each subject-object pair ID over all 70 predicates
    # rel_soP_prior = np.zeros((n_rel, self.dataset.n_pred))

    target = np.zeros((n_rel, self.n_pred))

    for i_rel,rel in enumerate(rels):

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

    yield img_blob
    yield spatial_features
    yield semantic_features
    yield target

if __name__ == '__main__':
  pass
