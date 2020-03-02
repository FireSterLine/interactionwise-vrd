# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import os.path as osp
import sys
this_dir = osp.dirname(__file__)
sys.path.insert(0, osp.join(this_dir, "faster-rcnn", "lib"))
import numpy as np
import argparse
import pprint

import time
import cv2
import torch
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms
import torchvision.datasets as dset

from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.rpn.bbox_transform import clip_boxes
#from model.nms.nms_wrapper import nms
from model.roi_layers import nms
from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.net_utils import save_net, load_net, vis_detections
from model.utils.blob import im_list_to_blob
from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet
from easydict import EasyDict

from lib.dataset import dataset
import globals
import utils
import pdb

# This class is the wrapper for the Object Detection model
#  It currently only supports Faster RCNN, but it can be expanded

class obj_detector():

  def __init__(self, dataset_name="vg", net="res101", pretrained="faster_rcnn_1_20_16193.pth"):

    print("obj_detector() called with args:")
    print([dataset_name, net, pretrained])

    self.dataset_name = dataset_name
    self.net = net
    self.pretrained = pretrained

    self.cuda = torch.cuda.is_available()
    # self.cuda = True ...
    self.class_agnostic = False
    self.dataset = dataset(self.dataset_name, with_bg_obj=True)

    # If a config file is specified for the model, then it is loaded, and the default model configurations are overridden
    # Those model parameters which are not specified in this config file, the default parameter values are used for them
    # If no config file is specified, then simply the default parameters are used, which are specified in models.utils.config
    cfg_from_file(osp.join(globals.faster_rcnn_dir, "cfgs", "{}.yml").format(self.net))

    cfg.USE_GPU_NMS = self.cuda

    print("Using config:")
    pprint.pprint(cfg)

    np.random.seed(cfg.RNG_SEED)

    lr = cfg.TRAIN.LEARNING_RATE
    momentum = cfg.TRAIN.MOMENTUM
    weight_decay = cfg.TRAIN.WEIGHT_DECAY

    print("Creating detector...")

    load_pretrained = isinstance(self.pretrained, str)

    # Create Faster R-CNN architecture
    # Pretrained = False forbids to load the toddler net, which is not needed since
    # we load_state_dict the full fasterRCNN architecture a few lines later.
    # initilize the network here.
    if self.net == "vgg16":
      self.fasterRCNN = vgg16(self.dataset.obj_classes, pretrained = not load_pretrained, class_agnostic = self.class_agnostic)
    elif self.net == "res101":
      self.fasterRCNN = resnet(self.dataset.obj_classes, 101, pretrained = not load_pretrained, class_agnostic = self.class_agnostic)
    elif self.net == "res50":
      self.fasterRCNN = resnet(self.dataset.obj_classes, 50, pretrained = not load_pretrained, class_agnostic = self.class_agnostic)
    elif self.net == "res152":
      self.fasterRCNN = resnet(self.dataset.obj_classes, 152, pretrained = not load_pretrained, class_agnostic = self.class_agnostic)
    else:
      raise Exception("Couldn't load unknown network: {}".format(self.net))

    self.fasterRCNN.create_architecture()

    if load_pretrained:
      model_path = osp.join(globals.faster_rcnn_models_dir, self.net, self.dataset_name, self.pretrained)

      print("Loading model... (checkpoint {})".format(model_path))

      if not osp.isfile(model_path):
        raise Exception("Pretrained model not found: {}".format(model_path))

      if self.cuda > 0:
        checkpoint = torch.load(model_path)
      else:
        checkpoint = torch.load(model_path, map_location=(lambda storage, loc: storage))
      self.fasterRCNN.load_state_dict(checkpoint["model"])

    # pooling_mode can be one of three types: align, crop or pool.
    # This parameter defines the kind of ROI applied to obtain the pooled features
    if "pooling_mode" in checkpoint.keys():
      cfg.POOLING_MODE = checkpoint["pooling_mode"]

    # initilize the tensor holder here.
    self.im_data   = torch.FloatTensor(1)
    self.im_info   = torch.FloatTensor(1)
    self.num_boxes = torch.LongTensor(1)
    self.gt_boxes  = torch.FloatTensor(1)

    # ship to cuda
    if self.cuda > 0:
      self.im_data   = self.im_data.cuda()
      self.im_info   = self.im_info.cuda()
      self.num_boxes = self.num_boxes.cuda()
      self.gt_boxes  = self.gt_boxes.cuda()

    cfg.CUDA = True

    if self.cuda > 0:
      self.fasterRCNN.cuda()

    # Set the model to evaluation mode
    if self.cuda > 0:
      self.fasterRCNN.eval()

  # This function generates, given an input image,
  #  the image pyramid which is then input to the network
  def _get_image_blob(self, im_bgr):
    """Converts an image into a network input.
    Arguments:
      im_bgr (ndarray): a color image in BGR order
    Returns:
      blob (ndarray): a data blob holding an image pyramid
      im_scale_factors (list): list of image scales (relative to im_bgr) used
        in the image pyramid
    """
    im_orig = im_bgr.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    # Generate pyramis as a list of images
    processed_ims = []
    im_scale_factors = []

    for target_size in cfg.TEST.SCALES:
      im_scale = float(target_size) / float(im_size_min)
      # Prevent the biggest axis from being more than MAX_SIZE
      if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
        im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
      im_bgr = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                interpolation=cv2.INTER_LINEAR)
      im_scale_factors.append(im_scale)
      processed_ims.append(im_bgr)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, np.array(im_scale_factors)

  # Similar to vis_detections in faster-rcnn/lib/model/utils/net_utils.py
  def res_detections(self, im, class_ix, class_name, dets, res, thresh=0.8):
    """Visual debugging of detections."""
    for i in range(np.minimum(10, dets.shape[0])):
      bbox = tuple(int(np.round(x)) for x in dets[i, :4])
      score = dets[i, -1]
      if score > thresh:
        cv2.rectangle(im, bbox[0:2], bbox[2:4], (0, 204, 0), 2)
        cv2.putText(im, "%s%d: %.3f" % (class_name, len(res["cls"]), score), (bbox[0], bbox[1] + 15), cv2.FONT_HERSHEY_PLAIN,
                    2.0, (0, 0, 255), thickness=2)
        res["box"] = np.vstack((res["box"], dets[i, :4]))
        res["cls"].append(class_ix-1)
        res["confs"].append(score)
    return im

  def det_im(self, im_file, show = False):

    print("det_im({})".format(im_file))

    thresh = 0.05

    total_tic = time.time()
    im_in = utils.read_img(im_file)

    # This probably allows black and white images
    if len(im_in.shape) == 2:
      im_in = im_in[:,:,np.newaxis]
      im_in = np.concatenate((im_in,im_in,im_in), axis=2)

    # RGB -> BGR
    im_bgr = im_in[:,:,::-1]

    blobs, im_scales = self._get_image_blob(im_bgr)

    assert len(im_scales) == 1, "Only single-image batch implemented"
    im_blob = blobs
    im_info_np = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)

    im_data_pt = torch.from_numpy(im_blob)
    im_data_pt = im_data_pt.permute(0, 3, 1, 2)
    im_info_pt = torch.from_numpy(im_info_np)

    with torch.no_grad():
      self.im_data.resize_(im_data_pt.size()).copy_(im_data_pt)
      self.im_info.resize_(im_info_pt.size()).copy_(im_info_pt)
      self.gt_boxes.resize_(1, 1, 5).zero_()
      self.num_boxes.resize_(1).zero_()

    det_tic = time.time()

    rois, cls_prob, bbox_pred, \
    rpn_loss_cls, rpn_loss_box, \
    RCNN_loss_cls, RCNN_loss_bbox, \
    rois_label = self.fasterRCNN(self.im_data, self.im_info, self.gt_boxes, self.num_boxes)

    scores = cls_prob.data
    boxes = rois.data[:, :, 1:5]
    # pdb.set_trace()

    # This box regression seems to post-process boxes (in order to remove useless ones?)
    if cfg.TEST.BBOX_REG:
      # Apply bounding-box regression deltas
      box_deltas = bbox_pred.data
      if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
        # Optionally normalize targets by a precomputed mean and stdev
        if self.cuda > 0:
          box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                     + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
        else:
          box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS) \
                     + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)

        if self.class_agnostic:
          box_deltas = box_deltas.view(1, -1, 4)
        else:
          box_deltas = box_deltas.view(1, -1, 4 * len(self.dataset.obj_classes))

      pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)

      # Clip boxes to image size
      pred_boxes = clip_boxes(pred_boxes, self.im_info.data, 1)
    else:
      # Simply repeat the boxes, once for each class
      # pred_boxes = torch.FloatTensor(np.tile(boxes.data.cpu().numpy(), (1, scores.shape[1]))).cuda()
      pred_boxes = np.tile(boxes, (1, scores.shape[1]))

    pred_boxes /= im_scales[0]

    scores = scores.squeeze()
    pred_boxes = pred_boxes.squeeze()
    det_toc = time.time()
    detect_time = det_toc - det_tic
    misc_tic = time.time()

    im2show = np.copy(im_in)

    res = {}
    res["box"] = np.zeros((0,4))
    res["cls"] = []
    res["confs"] = []

    for j in range(1, len(self.dataset.obj_classes)):
      inds = torch.nonzero(scores[:,j]>thresh).view(-1)
      # if there is det
      if inds.numel() > 0:
        cls_scores = scores[:,j][inds]
        _, order = torch.sort(cls_scores, 0, True)
        if self.class_agnostic:
          cls_boxes = pred_boxes[inds, :]
        else:
          cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]

        cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
        # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
        cls_dets = cls_dets[order]
        # keep = nms(cls_dets, cfg.TEST.NMS, force_cpu=not cfg.USE_GPU_NMS)
        keep = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS)
        cls_dets = cls_dets[keep.view(-1).long()]

        # im2show = vis_detections(im2show, self.dataset.obj_classes[j], cls_dets.cpu().numpy(), 0.5)
        im2show = self.res_detections(im2show, j, self.dataset.obj_classes[j], cls_dets.cpu().numpy(), res, 0.5)

    misc_toc = time.time()
    nms_time = misc_toc - misc_tic

    result_path = osp.join(globals.images_det_dir, im_file[:-4] + ".jpg")
    cv2.imwrite(result_path, im2show)

    sys.stdout.write("im_detect: {:.3f}s {:.3f}s   \r".format(detect_time, nms_time))
    sys.stdout.flush()

    cv2.imshow("Test detection", im2show)
    cv2.waitKey(0)

    return res

if __name__ == "__main__":
  det = obj_detector()

  print("Loading test images...")

  imglist = os.listdir(globals.images_dir)
  num_images = len(imglist)

  print("Loaded {} images.".format(num_images))

  for img_path in imglist:
    det.det_im(osp.join(globals.images_dir, img_path), True)
