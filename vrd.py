#from __future__ import absolute_import

import os
import os.path as osp
import sys
import time
import pickle
import argparse

import torch
import torch.nn as nn
import torch.nn.init
import cv2
import numpy as np

import sys

from lib.nets.vrd_model import vrd_model
from lib.blob import prep_im_for_blob
from easydict import EasyDict

from obj_det import obj_detector
from lib.dataset import dataset
import common
import pdb

class vr_detector():

  def __init__(self, dataset_name="vg", pretrained="epoch_4_checkpoint.pth(.tar)"):

    print("vr_detector() called with args:")
    print({dataset_name, pretrained})

    self.dataset_name = dataset_name
    self.pretrained = False # TODO

    print("Initializing object detector...")
    self.obj_det = obj_detector(dataset_name=self.dataset_name)

    self.dataset = dataset(self.dataset_name, with_bg_obj=True)

    self.N = self.dataset.n_obj
    self.M = self.dataset.n_pred

    self.args = EasyDict()
    self.args.dataset = self.dataset_name
    # this decides whether we are using visual features of the subject and object individually as well
    # or not, along with the visual features of the box bounding the two objects
    self.args.use_so = True
    # this decides whether we use the embeddings of the objects or not
    self.args.use_obj = True
    self.args.no_obj_prior = True
    # we have one of two location type features here; we can choose between them through 1 and 2
    # if this is set to 0, it means we are not using spatial features
    self.args.loc_type = 0

    # so_prior is a N*M*N dimension array, which contains the prior probability distribution of
    # each object pair over all 70 predicates. This was calculated beforehand, probably from the
    # co-occurance of predicates with respect to subject-object pairs
    with open("data/{}/so_prior.pkl".format(dataset_name), 'rb') as fid:
      self.so_prior = cPickle.load(fid)


    load_pretrained = isinstance(self.pretrained, str)

    # initialize the model using the args set above
    print("Initializing VRD Model...")
    self.net = vrd_model(self.args) # TODO: load_pretrained affects how the model is initialized?
    self.net.cuda()
    self.net.eval()

    if load_pretrained:
      model_path = osp.join(common.models_dir, self.pretrained)

      print("Loading model... (checkpoint {})".format(model_path))

      if not osp.isfile(model_path):
        raise Exception("Pretrained model not found: {}".format(model_path))

      checkpoint = torch.load(model_path)
      self.net.load_state_dict(checkpoint["state_dict"])

  def det_im(self, im_path, objd_res):

    print("Detecting object...")
    objd_res = self.obj_det.det_im(im_path)

    # Read object detections
    boxes_img = objd_res["box"]
    pred_cls_img = np.array(objd_res["cls"])
    pred_confs = np.array(objd_res["confs"])

    time1 = time.time()

    im = common.read_img(im_path)
    ih = im.shape[0]
    iw = im.shape[1]

    PIXEL_MEANS = common.vrd_pixel_means
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
    rel_so_prior = np.zeros((n_rel_inst, self.dataset.n_pred))
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
        # store the probability distribution of this subject-object pair from the so_prior
        rel_so_prior[i_rel_inst] = self.so_prior[classes[s_idx], classes[o_idx]]
        i_rel_inst += 1
    boxes = boxes.astype(np.float32, copy=False)
    classes = classes.astype(np.float32, copy=False)
    ix1 = np.array(ix1)
    ix2 = np.array(ix2)

    # apply the VRD model on the pairs of objects
    # Here, obj_score contains the scores (not probabilities!) of all objects. Size = self.dataset.n_obj
    # Here, rel_score contains the scores (not probabilities!) of all predicates. Size = num_predicates
    obj_score, rel_score = self.net(blob, boxes, rel_boxes, SpatialFea, classes, ix1, ix2, self.args)
    # To figure out: why are these now probabilities and not scores anymore?
    rel_prob = rel_score.data.cpu().numpy()
    # Here, we are factoring in the prior probability using rel_so_prior
    rel_prob += np.log(0.5 * (rel_so_prior + 1.0 / self.dataset.n_pred))
    # this structure is for holding the subject class ID, subject ID, predicate ID, object class ID and object ID
    rlp_labels_im = np.zeros((rel_prob.shape[0] * rel_prob.shape[1], 5), dtype=np.int)
    # this is for storing the confidence scores. To figure out: Why is this named tuple?
    tuple_confs_im = []
    n_idx = 0
    for tuple_idx in range(rel_prob.shape[0]):
      sub = ix1[tuple_idx]
      obj = ix2[tuple_idx]
      for rel in range(rel_prob.shape[1]):
        # get the confidence score. To figure out: Does this mean it's not a probability?
        conf = rel_prob[tuple_idx, rel]
        # classes[sub] and classes[obj] will give the actual class IDs
        # sub and rel are the subject and object IDs assigned as per their order of detection in the
        # original image. Therefore, the first object detected has ID 0, the second has ID 1, and so on
        rlp_labels_im[n_idx] = [classes[sub], sub, rel, classes[obj], obj]
        tuple_confs_im.append(conf)
        n_idx += 1
    tuple_confs_im = np.array(tuple_confs_im)
    # sort the confidence scores, reverse the sorting to get descending order, select the first 20,
    # and get their indices
    idx_order = tuple_confs_im.argsort()[::-1][:20]
    rlp_labels_im = rlp_labels_im[idx_order, :]
    tuple_confs_im = tuple_confs_im[idx_order]
    # this is for storing the final subject-predicate-object triples, and the corresponding confidence scores
    vrd_res = []
    for tuple_idx in range(rlp_labels_im.shape[0]):
      label_tuple = rlp_labels_im[tuple_idx]
      # get the class label of the subject
      # label_tuple[0] contains the class ID of the subject
      sub_cls = self.dataset.obj_classes[label_tuple[0]]
      # get the class label of the object
      # label_tuple[3] contains the class ID of the object
      obj_cls = self.dataset.obj_classes[label_tuple[3]]
      # get the class label of the predicate
      # label_tuple[2] contains the class ID of the predicate
      rel_cls = self.dataset.pred_classes[label_tuple[2]]
      vrd_res.append(("%s%d-%s-%s%d" % (sub_cls, label_tuple[1], rel_cls, obj_cls, label_tuple[4]), tuple_confs_im[tuple_idx]))
    print(vrd_res)

    time2 = time.time()

    print("TEST Time:%s" % (time.strftime("%H:%M:%S", time.gmtime(int(time2 - time1)))))

    return vrd_res

  # Union box of two boxes
  def getUnionBBox(self, aBB, bBB, ih, iw, margin=10):
    return [max(0, min(aBB[0], bBB[0]) - margin),
            max(0, min(aBB[1], bBB[1]) - margin),
            min(iw, max(aBB[2], bBB[2]) + margin),
            min(ih, max(aBB[3], bBB[3]) + margin)]

  # Relative location spatial feature
  def getRelativeLoc(self, aBB, bBB):
    sx1, sy1, sx2, sy2 = aBB.astype(np.float32)
    ox1, oy1, ox2, oy2 = bBB.astype(np.float32)
    sw, sh, ow, oh = sx2 - sx1, sy2 - sy1, ox2 - ox1, oy2 - oy1
    xy = np.array([(sx1 - ox1) / ow, (sy1 - oy1) / oh, (ox1 - sx1) / sw, (oy1 - sy1) / sh])
    wh = np.log(np.array([sw / ow, sh / oh, ow / sw, oh / sh]))
    return np.hstack((xy, wh))


def vrd_demo():

  # im_path = osp.join(common.images_dir, "3845770407_1a8cd41230_b.jpg")
  im_path = osp.join(common.images_dir, "img1.jpg")

  print("Initializing VRD module...")
  vr_det = vr_detector()

  print("Calling det_im for relationship detection...")
  vrd_res = vr_det.det_im(im_path)
  print(vrd_res)


if __name__ == '__main__':
  vrd_demo()
  #from IPython import embed; embed()
