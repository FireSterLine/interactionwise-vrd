import numpy as np
import os.path as osp
import scipy.io as sio
import scipy
import cv2
import json
import pickle
import sys
import math
import os.path as osp
from collections import defaultdict
import utils, globals
import warnings

# TODO: rename to VRDDataset
# TODO: add flag that forbids/allows caching with pickles
#  (default behaviour would be to pickle everything, since the dataset won't change that much)

class dataset():

  def __init__(self, name, subset=None, with_bg_obj=True, with_bg_pred=False, justafew=False):

    self.name         = name
    self.subset       = subset
    self.with_bg_obj  = with_bg_obj
    self.with_bg_pred = with_bg_pred
    self.justafew     = justafew

    if self.justafew:
      warnings.warn("Warning: Using less data (because of 'justafew' debugging)", UserWarning)

    self.img_dir      = None
    self.metadata_dir = None

    self._vrd_data_cache = {}

    if self.name == "vrd":
      self.img_dir = osp.join(globals.data_dir, "vrd", "sg_dataset")
      self.metadata_dir = osp.join(globals.data_dir, "vrd")

    elif self.name == "vg":

      if self.subset == None:
        self.subset = "1600-400-20"
      # self.subset = "2500-1000-500"
      # self.subset = "150-50-50"

      self.img_dir = osp.join(globals.data_dir, "vg")
      self.metadata_dir = osp.join(globals.data_dir, "genome", self.subset)

    else:
      raise Exception("Unknown dataset: {}".format(self.name))

    # load the vocabularies for objects and predicates
    with open(osp.join(self.metadata_dir, "objects.json"), 'r') as rfile:
      obj_classes = json.load(rfile)

    with open(osp.join(self.metadata_dir, "predicates.json"), 'r') as rfile:
      pred_classes = json.load(rfile)


    if with_bg_obj:
        obj_additional = np.asarray(["__background__"])
    else:
        obj_additional = np.asarray([])

    if with_bg_pred:
        pred_additional = np.asarray(["__nopredicate__"])
    else:
        pred_additional = np.asarray([])

    self.obj_classes = np.append(obj_classes, obj_additional).tolist()
    self.n_obj = len(self.obj_classes)

    self.pred_classes = np.append(pred_classes, pred_additional).tolist()
    self.n_pred = len(self.pred_classes)

    # Need these? Or use utils.invert_dict, fra
    # self.class_to_ind     = dict(zip(self._classes, xrange(self._num_classes)))
    # self.relations_to_ind = dict(zip(self._relations, xrange(self._num_relations)))

  def readImg(self, img_path):
    return utils.read_img(osp.join(self.img_dir, img_path))

  # Need alias for getRelst? Here we go: def getRelst(self, stage, granularity = "img"): return self.getData("relst", stage, granularity)

  def getData(self, format, stage, granularity = "img"):
    # TODO: figure out if we need annos for granularity = "rel"
    """ Load list of relationships """
    # print((format, stage, granularity))
    if not (format, stage, granularity) in self._vrd_data_cache:
      filename = "data_{}_{}_{}.json".format(format, granularity, stage)
      # ...filename = "dsr_{}_{}_{}.json".format(format, granularity, stage))
      print("Data not cached. Reading {}...".format(filename))
      with open(osp.join(self.metadata_dir, filename), 'r') as rfile:
        data = json.load(rfile)[0:3] if self.justafew else json.load(rfile)
        self._vrd_data_cache[(format, stage, granularity)] = data
    return self._vrd_data_cache[(format, stage, granularity)]
    # Annos:
    # with open(osp.join(globals.metadata_dir, "annos.pkl", 'rb') as fid:
    #   annos = pickle.load(fid)
    #   self._annos = [x for x in annos if x is not None and len(x['classes'])>1]

  def getDistribution(self, type, stage = "train"):
    """ Computes and returns some distributional data """

    if stage == "test":
        raise ValueError("Can't compute distribution on \"{}\" split".format(stage))

    distribution_pkl_path = osp.join(self.metadata_dir, "{}.pkl".format(type))
    distribution = None

    if type == "soP":
      assert stage == "train", "Wait a second, why do you want the soP for the train split?"
      try:
        # TODO: use our soP prior: raise FileNotFoundError
        # with open(distribution_pkl_path, 'rb') as fid:
        with open(osp.join(self.metadata_dir, "so_prior.pkl"), 'rb') as fid:
          print("Distribution {} found!".format(type))
          distribution = pickle.load(fid, encoding='latin1')
      except FileNotFoundError:
        print("Distribution {} not found: {}. Generating...".format(type, distribution_pkl_path))
        distribution = self._generate_soP_distr(self.getData("relst", stage))
        pickle.dump(distribution, open(distribution_pkl_path, 'wb'))
    else:
      raise Exception("Unknown distribution requested: {}".format(type))

    return distribution

  def _generate_soP_distr(self, relst):

    sop_counts = np.zeros((self.n_obj, self.n_obj, self.n_pred))

    # Count sop occurrences
    for img_path,rels in relst:
      if rels == None:
        continue
      for elem in rels:
        subject_label   = elem["subject"]["id"]
        object_label    = elem["object"]["id"]
        predicate_label = elem["predicate"]["id"]

        sop_counts[subject_label][object_label][predicate_label] += 1

    # Divide each line by # of counts
    for sub_idx in range(self.n_obj):
      for obj_idx in range(self.n_obj):
        total_count = sop_counts[sub_idx][obj_idx].sum()
        if total_count == 0:
          continue
        sop_counts[sub_idx][obj_idx] /= float(total_count)

    return sop_counts

  # TODO
  # def readMetadata(self, data_name):
  #   """ Wrapper for read/cache metadata file. This prevents loading the same metadata file more than once """
  #
  #   if not hasattr(self, data_name):
  #     with open(osp.join(self.metadata_dir, "{}.json".format(data_name)), 'r') as rfile:
  #       setattr(self, data_name, json.load(rfile))
  #
  #   return getattr(self, data_name)
