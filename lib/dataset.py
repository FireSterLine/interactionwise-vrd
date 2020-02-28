import numpy as np
import os.path as osp
import scipy.io as sio
import scipy
import cv2
import pickle
import sys
import math

class dataset():

  def __init__(self, name="vg", subset=None, with_bg_obj=True, with_bg_pred=False):

    self.name = "vg"
    self.subset = None
    self.with_bg_obj = True
    self.with_bg_pred = False

    if self.name == "pascal_voc":
      obj_classes = np.asarray(["aeroplane", "bicycle", "bird", "boat",
                         "bottle", "bus", "car", "cat", "chair",
                         "cow", "diningtable", "dog", "horse",
                         "motorbike", "person", "pottedplant",
                         "sheep", "sofa", "train", "tvmonitor"])
    elif self.name == "vg":
      if self.subset == None:
        self.subset = "1600-400-20"
      # self.subset = "2500-1000-500"
      # self.subset = "150-50-50"
      with open("data/genome/{}/objects_vocab.txt".format(self.subset), 'r') as f:
        obj_vocab = f.readlines()
        obj_classes = np.asarray([x.strip('\n') for x in obj_vocab])

      with open("data/genome/{}/relations_vocab.txt".format(self.subset), 'r') as f:
        pred_vocab = f.readlines()
        pred_classes = np.asarray([x.strip('\n') for x in pred_vocab])
    else:
      raise Exception("Unknown dataset: " + self.name)

    if with_bg_obj:
        obj_additional = np.asarray(["__background__"])
    else:
        obj_additional = np.asarray([])

    if with_bg_pred:
        pred_additional = np.asarray(["__nopredicate__"])
    else:
        pred_additional = np.asarray([])

    self.obj_classes = np.append(obj_additional, obj_classes).tolist()
    self.n_obj = len(self.obj_classes)

    self.pred_classes = np.append(pred_additional, pred_classes).tolist()
    self.n_pred = len(self.pred_classes)
