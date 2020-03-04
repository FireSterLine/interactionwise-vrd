#from __future__ import absolute_import

import os
import os.path as osp
import sys
import time
import pickle
import argparse

import cv2
import torch
import torch.nn as nn
import torch.nn.init
import numpy as np
from gensim.models import KeyedVectors

import sys

from lib.nets.vrd_model import vrd_model
from lib.blob import prep_im_for_blob
from easydict import EasyDict

from obj_det import obj_detector
from lib.dataset import dataset
import globals
import utils
import pdb

class vr_detector():

  def __init__(self, dataset_name="vg"): # , pretrained=False): pretrained="epoch_4_checkpoint.pth(.tar)"):

    print("vr_detector() called with args:")
    print([dataset_name, pretrained])

    self.dataset_name = dataset_name
    self.pretrained = False # TODO

    print("Initializing object detector...")
    self.obj_det = obj_detector(dataset_name=self.dataset_name)

    self.dataset = dataset(self.dataset_name, with_bg_obj=True)

    self.args = EasyDict()
    self.args.n_obj   = self.dataset.n_obj
    self.args.n_pred  = self.dataset.n_pred

    # soP_prior is a N*M*N dimension array, which contains the prior probability distribution of
    #  each object pair over all 70 predicates.
    self.soP_prior = self.dataset.getDistribution("soP", True)

    load_pretrained = isinstance(self.pretrained, str)

    print("Loading Word2Vec model...")
    self.w2v_model = KeyedVectors.load_word2vec_format(globals.w2v_model_path, binary=True)

    # initialize the model using the args set above
    print("Initializing VRD Model...")
    self.net = vrd_model(self.args) # TODO: load_pretrained affects how the model is initialized?
    self.net.cuda()
    self.net.eval()

    if load_pretrained:
      model_path = osp.join(globals.models_dir, self.pretrained)

      print("Loading model... (checkpoint {})".format(model_path))

      if not osp.isfile(model_path):
        raise Exception("Pretrained model not found: {}".format(model_path))

      checkpoint = torch.load(model_path)
      self.net.load_state_dict(checkpoint["state_dict"])

  def test_vrd_model(self):
    img_rels = self.dataset.getImgRels()

    for im_id,rels in img_rels.items():

      print("IMG: {}".format(osp.join(self.dataset.img_dir, im_id)))

      im = utils.read_img(osp.join(self.dataset.img_dir, im_id))
      ih = im.shape[0]
      iw = im.shape[1]

      print(im.shape)

      n_rel = len(rels)

      print(n_rel)

      # the dimension 8 here is the size of the spatial feature vector, containing the relative location and log-distance
      spatial_features = np.zeros((n_rel, 8))
      # the dimension 8 here is the size of the spatial feature vector, containing the relative location and log-distance
      semantic_features = np.zeros((n_rel, 2*300))
      # this will contain the probability distribution of each subject-object pair ID over all 70 predicates
      rel_soP_prior = np.zeros((n_rel, self.dataset.n_pred))

      for i_rel,rel in enumerate(rels):

        print(i_rel,rel)

        # these are the subject and object bounding boxes
        sBBox = utils.bboxDictToNumpy(rel["subject"]["bbox"])
        oBBox = utils.bboxDictToNumpy(rel["object"]["bbox"])

        # store the scaled dimensions of the union bounding box here, with the id i_rel
        spatial_features[i_rel] = utils.getRelativeLoc(sBBox, oBBox)

        # semantic features of obj and subj
        semantic_features[i_rel] = utils.getSemanticVector(rel['subject']['name'], rel['object']['name'], self.w2v_model)
        # semantic_features[i_rel] = np.zeros(600)

        # store the probability distribution of this subject-object pair from the soP_prior
        if self.soP_prior != None:
          rel_soP_prior[i_rel] = self.soP_prior[classes[s_idx], classes[o_idx]]
        else:
          rel_soP_prior[i_rel,:] = 0.0
        i_rel += 1

      print(spatial_features)
      print(semantic_features)

      spatial_features  = torch.FloatTensor(spatial_features).cuda()
      semantic_features = torch.FloatTensor(semantic_features).cuda()

      x = self.net(spatial_features, semantic_features)
      print(x)

      break

  def det_im(self, im_path):

    print("Detecting object...")
    objd_res = self.obj_det.det_im(im_path)

    # Read object detections
    boxes_img = objd_res["box"]
    pred_cls_img = np.array(objd_res["cls"])
    pred_confs = np.array(objd_res["confs"])

    time1 = time.time()

    im = utils.read_img(im_path)
    ih = im.shape[0]
    iw = im.shape[1]

    PIXEL_MEANS = globals.vrd_pixel_means
    image_blob, im_scale = prep_im_for_blob(im, PIXEL_MEANS)

    blob = np.zeros((1,) + image_blob.shape, dtype=np.float32)
    blob[0] = image_blob

    # Reshape net's input blobs
    # boxes holds the scaled dimensions of the object boxes.
    boxes = np.zeros((boxes_img.shape[0], 5))
    # These dimensions are in indices 1 to 5
    boxes[:, 1:5] = boxes_img * im_scale
    classes = pred_cls_img

    ix1 = []
    ix2 = []
    # the total number of union bounding boxes is n(n-1),
    # where n is the number of objects identified in the image, or pred_cls_img
    n_rel_inst = len(pred_cls_img) * (len(pred_cls_img) - 1)
    # rel_boxes contains the scaled dimensions of the union bounding boxes
    rel_boxes = np.zeros((n_rel_inst, 5))
    # the dimension 8 here is the size of the spatial feature vector, containing the relative location and log-distance
    SpatialFea = np.zeros((n_rel_inst, 8))
    # this will contain the probability distribution of each subject-object pair ID over all 70 predicates
    rel_soP_prior = np.zeros((n_rel_inst, self.dataset.n_pred))
    # this is used as an ID for each subject-object pair; it increments at the end of the inner loop below

    i_rel_inst = 0
    for s_idx in range(len(pred_cls_img)):
      for o_idx in range(len(pred_cls_img)):
        # if the object is the same as itself, skip it
        if(s_idx == o_idx):
            continue
        ix1.append(s_idx)
        ix2.append(o_idx)
        # these are the subject and object bounding boxes
        sBBox = boxes_img[s_idx]
        oBBox = boxes_img[o_idx]
        # get the union bounding box
        rBBox = self.getUnionBBox(sBBox, oBBox, ih, iw)
        # store the scaled dimensions of the union bounding box here, with the id i_rel_inst
        rel_boxes[i_rel_inst, 1:5] = np.array(rBBox) * im_scale
        SpatialFea[i_rel_inst] = self.getRelativeLoc(sBBox, oBBox)
        # store the probability distribution of this subject-object pair from the soP_prior
        if self.soP_prior != None:
          rel_soP_prior[i_rel_inst] = self.soP_prior[classes[s_idx], classes[o_idx]]
        else:
          rel_soP_prior[i_rel_inst,:] = 0.0
        i_rel_inst += 1
    boxes = boxes.astype(np.float32, copy=False)
    classes = classes.astype(np.float32, copy=False)
    ix1 = np.array(ix1)
    ix2 = np.array(ix2)

    x = self.net(blob, boxes, rel_boxes, SpatialFea, classes, ix1, ix2, self.args)

    return x

def vrd_demo():

  # im_path = osp.join(globals.images_dir, "3845770407_1a8cd41230_b.jpg")
  im_path = osp.join("img-vrd", "3845770407_1a8cd41230_b.jpg")

  print("Initializing VRD module...")
  vr_det = vr_detector()

  print("Calling det_im for relationship detection...")
  vrd_res = vr_det.det_im(im_path)
  print(vrd_res)


if __name__ == '__main__':
  print("Initializing VRD module...")
  vr_det = vr_detector()

  vr_det.test_vrd_model()

  # vrd_demo()
  #from IPython import embed; embed()
