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
import globals

# TODO: rename to VRDDataset
# TODO: add flag that forbids/allows caching with pickles
#  (default behaviour would be to pickle everything, since the dataset won't change that much)

class dataset():

  def __init__(self, name="vg", subset=None, with_bg_obj=True, with_bg_pred=False):

    self.name = name
    self.subset = None
    self.with_bg_obj = True
    self.with_bg_pred = False

    self.img_dir = None
    self.metadata_dir = None

    if self.name == "vrd":
      self.img_dir = osp.join(globals.data_dir, "vrd", "sg_dataset")
      self.metadata_dir = osp.join(globals.data_dir, "vrd")

      # load the vocabularies for objects and predicates
      with open(osp.join(self.metadata_dir, "objects.json"), 'r') as rfile:
        obj_classes = json.load(rfile)

      with open(osp.join(self.metadata_dir, "predicates.json"), 'r') as rfile:
        pred_classes = json.load(rfile)

    elif self.name == "vg":

      if self.subset == None:
        self.subset = "1600-400-20"
      # self.subset = "2500-1000-500"
      # self.subset = "150-50-50"

      self.img_dir = osp.join(globals.data_dir, "vg")
      self.metadata_dir = osp.join(globals.data_dir, "genome", self.subset)

      # load the vocabularies for objects and predicates
      with open(osp.join(self.metadata_dir, "objects_vocab.txt"), 'r') as f:
        obj_vocab = f.readlines()
        obj_classes = np.asarray([x.strip('\n') for x in obj_vocab])

      with open(osp.join(self.metadata_dir, "relations_vocab.txt"), 'r') as f:
        pred_vocab = f.readlines()
        pred_classes = np.asarray([x.strip('\n') for x in pred_vocab])

    else:
      raise Exception("Unknown dataset: {}".format(self.name))

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

    # Need these?
    # self.class_to_ind     = dict(zip(self._classes, xrange(self._num_classes)))
    # self.relations_to_ind = dict(zip(self._relations, xrange(self._num_relations)))


  # TODO: select which split ("train", "test", default="traintest")
  def getImgRels(self):
    """ Load list of rel-annotations per images """
    with open(osp.join(self.metadata_dir, "vrd_data.json"), 'r') as rfile:
      return json.load(rfile) # Maybe pickle this?

  def getAnno(self):
    """ Load list of  """
    pass
    # with open(osp.join(globals.metadata_dir, "anno.pkl", 'rb') as fid:
    #   anno = pickle.load(fid)
    #   self._anno = [x for x in anno if x is not None and len(x['classes'])>1]

  # Todo: instead of simply return it, store it in self and return a reference to, say, self.soP
  def getDistribution(self, type, force = True):
    """ Computes and returns some distributional data """

    if not type in ["soP"]:
      raise Exception("Unknown distribution requested: {}".format(type))

    distribution_pkl_path = osp.join(self.metadata_dir, "{}.pkl".format(type))

    # TODO: remove this fix once we have our own so_prior
    if type == "soP" and self.name == "vrd":
        distribution_pkl_path = osp.join(self.metadata_dir, "so_prior.pkl")

    try:
      with open(distribution_pkl_path, 'rb') as fid:
        return pickle.load(fid)
    except EnvironmentError: # parent of IOError, OSError *and* WindowsError where available
      if not force:
        print("Distribution not found: {}".format(distribution_pkl_path))
        return None
      else:
        raise Exception("TODO: compute distribution, save pkl and return it.")
