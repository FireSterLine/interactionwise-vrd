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

# TODO: add flag that forbids/allows caching with pickles
#  (default behaviour would be to pickle everything, since the dataset won't change that much)

class VRDDataset():

  def __init__(self, name, subset = None, with_bg_obj=True, with_bg_pred=False, justafew=False):

    img_subset = ""
    # This allows to use names like "vrd/dsr", "vg/150-50-50"
    if "/" in name and subset == None:
      pieces = name.split("/")
      if len(pieces) == 2:
        name,subset = pieces
      elif len(pieces) == 3:
        name,subset,img_subset = pieces
      else:
        raise ValueError("Can't initialize VRDDataset with {}".format(pieces))

    self.name         = name
    self.subset       = subset
    self.img_subset   = img_subset
    self.with_bg_obj  = with_bg_obj
    self.with_bg_pred = with_bg_pred
    self.justafew     = justafew

    if self.justafew != False:
      print("Warning! Using less data (because of 'justafew' debugging)")

    if self.img_subset != "" and self.name != "vg":
      raise ValueError("Couldn't initialize img_subset '{}' for dataset '{}'".format(self.img_subset, self.name))

    self.img_dir      = None
    self.metadata_dir = None

    self._vrd_data_cache = {}
    self._distr_cache    = {}

    if self.name == "vrd":
      self.img_dir = osp.join(globals.data_dir, "vrd", "sg_dataset")
      self.metadata_dir = osp.join(globals.data_dir, "vrd")

    elif self.name == "vg":

      if self.subset == None:
        # self.subset = "1600-400-20"
        # self.subset = "2500-1000-500"
        self.subset = "150-50-50"

      self.img_dir = osp.join(globals.data_dir, "vg")
      self.metadata_dir = osp.join(globals.data_dir, "genome", self.subset)

    else:
      raise Exception("Unknown dataset: {}".format(self.name))

    # load the vocabularies for objects and predicates
    obj_additional  = []
    pred_additional = []
    if with_bg_obj:  obj_additional  += ["__background__"]
    if with_bg_pred: pred_additional += ["__nopredicate__"]

    self.obj_classes = self.readJSON("objects.json") + obj_additional
    self.n_obj = len(self.obj_classes)

    self.pred_classes = self.readJSON("predicates.json") + pred_additional
    self.n_pred = len(self.pred_classes)

    # Need these? Or use utils.invert_dict, fra
    # self.obj_labels  = utils.invert_dict(self.obj_classes)
    # self.pred_labels = utils.invert_dict(self.pred_classes)

  def readImg(self, img_path):
    return utils.read_img(osp.join(self.img_dir, img_path))

  def getRelst(self, **kwargs): return self.getData("relst", **kwargs)
  def getAnnos(self, **kwargs): return self.getData("annos", **kwargs)

  def getData(self, format, stage, granularity = "img"):
    # TODO: figure out if we need annos for granularity = "rel"
    """ Load list of relationships """
    # print((format, stage, granularity))
    if not (format, stage, granularity) in self._vrd_data_cache:
      filename = "data_{}_{}_{}.json".format(format, granularity, stage)
      if self.name == "vrd" and self.subset == "dsr":
          filename = "dsr_{}_{}_{}.json".format(format, granularity, stage)
      # print("Data not cached. Reading {}...".format(filename))
      data = self.readJSON(filename)
      if self.img_subset != "":
        if self.img_subset == "mini" and stage == "train":
          data = data[:1000]
        elif self.img_subset == "small" and stage == "train":
          data = data[:20000]
        elif self.img_subset == "mini" and stage == "test":
          data = data[:100]
        elif self.img_subset == "small" and stage == "test":
          data = data[:2000]

      if self.justafew == True:
          data = data[:100]
      elif self.justafew != False and isinstance(self.justafew, int):
          data = data[self.justafew:self.justafew+1]
      self._vrd_data_cache[(format, stage, granularity)] = data
    return self._vrd_data_cache[(format, stage, granularity)]

  def getDistribution(self, type, stage = "train"):
    """ Computes and returns some distributional data """

    if stage != "train":
      raise ValueError("Can't compute distribution on \"{}\" split".format(stage))

    if not (type, stage) in self._distr_cache:
      distribution_pkl_path = osp.join(self.metadata_dir, "{}.pkl".format(type))
      dont_pkl = False
      distribution = None

      if type == "soP" and self.subset == "dsr":
        distribution_pkl_path = osp.join(self.metadata_dir, "so_prior.pkl")
        dont_pkl = True

      try:
        with open(distribution_pkl_path, 'rb') as fid:
          print("Distribution {} found!".format(type))
          distribution = pickle.load(fid, encoding='latin1')
      except FileNotFoundError:
        print("Distribution {} not found: {}. Generating...".format(type, distribution_pkl_path))
        if type == "soP":
          distribution = self._generate_soP_distr(self.getData("relst", stage))
        elif type == "pP":
          distribution = self._generate_pP_distr(self.getData("relst", stage))
        else:
          raise Exception("Unknown distribution requested: {}".format(type))

      if not dont_pkl:
        pickle.dump(distribution, open(distribution_pkl_path, 'wb'))
      self._distr_cache[(type, stage)] = distribution

    return self._distr_cache[(type, stage)]

  # Count sop-triples occurrences
  def _get_sop_counts(self, relst):
    sop_counts = np.zeros((self.n_obj, self.n_obj, self.n_pred))
    for img_path,rels in relst:
      if img_path == None:
        continue
      for elem in rels:
        subject_label    = elem["subject"]["id"]
        object_label     = elem["object"]["id"]
        predicate_labels = elem["predicate"]["id"]

        sop_counts[subject_label][object_label][predicate_labels] += 1

  def _generate_soP_distr(self, relst):
    sop_counts = self._get_sop_counts(relst)
    # Divide each line by # of counts
    for sub_idx in range(self.n_obj):
      for obj_idx in range(self.n_obj):
        total_count = sop_counts[sub_idx][obj_idx].sum()
        if total_count == 0:
          continue
        sop_counts[sub_idx][obj_idx] /= float(total_count)

    return sop_counts


  def _generate_pP_distr(self, relst):
    pp_counts = np.zeros((self.n_pred, self.n_pred))

    sop_counts = self._get_sop_counts(relst)
    raise NotImplementedError

    # Divide each line by # of counts
    for pred1_idx in range(self.n_pred):
      for pred2_idx in range(self.n_pred):
        total_count = pp_counts[pred1_idx][pred2_idx].sum()
        if total_count == 0:
          continue
        pp_counts[pred1_idx][pred2_idx] /= float(total_count)

    return pp_counts

  # Read json datafile
  def readJSON(self, filename):
    with open(osp.join(self.metadata_dir, filename), 'r') as rfile:
      return json.load(rfile)

  # Read pickle datafile
  def readPKL(self, filename):
    with open(osp.join(self.metadata_dir, filename), 'rb') as rfile:
      return pickle.load(rfile, encoding="latin1")
  # TODO
  # def readMetadata(self, data_name):
  #   """ Wrapper for read/cache metadata file. This prevents loading the same metadata file more than once """
  #   if not hasattr(self, data_name):
  #     with open(osp.join(self.metadata_dir, "{}.json".format(data_name)), 'r') as rfile:
  #       setattr(self, data_name, json.load(rfile))
  #
  #   return getattr(self, data_name)
