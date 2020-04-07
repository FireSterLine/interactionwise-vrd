import os
import cv2
import json
import numpy as np
import model.utils.net_utils as frcnn_net_utils
import time
from copy import deepcopy
import torch
import warnings
import globals
import munch

# TODO: figure out what pixel means to use, how to compute them:
#  do they come from the dataset used for training, perhaps?
# Read here: https://github.com/GriffinLiang/vrd-dsr/issues/12
# If you use the pretrained, you should use the same value. Boh
vrd_pixel_means = np.array([[[102.9801, 115.9465, 122.7717]]])

# Pytorch CUDA Fallback
# TODO: check if this works and then use it everywhere instead of cuda()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.LongTensor([1]).to(device) # Test


weights_normal_init  = frcnn_net_utils.weights_normal_init
save_checkpoint      = frcnn_net_utils.save_checkpoint
load_checkpoint      = lambda path : torch.load(path, map_location = device)
adjust_learning_rate = frcnn_net_utils.adjust_learning_rate


# Bbox as a list to numpy array
bboxListToNumpy = np.array

# Bbox as a dict to numpy array
def bboxDictToNumpy(bbox_dict):
  return bboxListToNumpy(bboxDictToList(bbox_dict))

# Bbox as a dict to list
def bboxDictToList(bbox_dict):
  return [bbox_dict["xmin"],
          bbox_dict["ymin"],
          bbox_dict["xmax"],
          bbox_dict["ymax"]]

# Union box of two boxes
def getUnionBBox(aBB, bBB, ih, iw, margin=10):
  return [max(0,  min(aBB[0], bBB[0]) - margin),
          max(0,  min(aBB[1], bBB[1]) - margin),
          min(iw, max(aBB[2], bBB[2]) + margin),
          min(ih, max(aBB[3], bBB[3]) + margin)]

# Relative location spatial feature
def getRelativeLoc(aBB, bBB):
  sx1, sy1, sx2, sy2 = aBB.astype(np.float32)
  ox1, oy1, ox2, oy2 = bBB.astype(np.float32)
  sw, sh, ow, oh = sx2 - sx1, sy2 - sy1, ox2 - ox1, oy2 - oy1
  xy = np.array([(sx1 - ox1) / ow, (sy1 - oy1) / oh, (ox1 - sx1) / sw, (oy1 - sy1) / sh])
  wh = np.log(np.array([sw / ow, sh / oh, ow / sw, oh / sh]))
  return np.hstack((xy, wh))

"""
def getDualMask(self, ih, iw, bb):
  rh = 32.0 / ih
  rw = 32.0 / iw
  x1 = max(0, int(math.floor(bb[0] * rw)))
  x2 = min(32, int(math.ceil(bb[2] * rw)))
  y1 = max(0, int(math.floor(bb[1] * rh)))
  y2 = min(32, int(math.ceil(bb[3] * rh)))
  mask = np.zeros((32, 32))
  mask[y1 : y2, x1 : x2] = 1
  assert(mask.sum() == (y2 - y1) * (x2 - x1))
  return mask
"""

def getEmbedding(word, emb_model, depth=0):
    if not hasattr(getEmbedding, "fallback_emb_map"):
        # This map defines the fall-back words of words that do not exist in the embedding model
        with open(os.path.join(globals.data_dir, "embeddings", "fallback-v1.json"), 'r') as rfile:
            getEmbedding.fallback_emb_map = json.load(rfile)
    try:
        embedding = emb_model[word]
    except KeyError:
        embedding = np.zeros(300)
        fallback_words = []
        if word in getEmbedding.fallback_emb_map:
          fallback_words = getEmbedding.fallback_emb_map[word]
        if " " in word:
          fallback_words = ["_".join(word.split(" "))] + fallback_words + [word.split(" ")]

        for fallback_word in fallback_words:
          if isinstance(fallback_word, str):
            embedding = getEmbedding(fallback_word, emb_model, depth+1)
            if np.all(embedding != np.zeros(300)):
              print("{}'{}' mapped to '{}'".format("  " * depth, word, fallback_word))
              break
          elif isinstance(fallback_word, list):
            fallback_vec = [getEmbedding(fb_sw, emb_model, depth+1) for fb_sw in fallback_word]
            filtered_wv = [(w,v) for w,v in zip(fallback_word,fallback_vec) if not np.all(v == np.zeros(300))]
            fallback_w,fallback_v = [],[]
            if len(filtered_wv) > 0:
              fallback_w,fallback_v = zip(*filtered_wv)
              embedding = np.mean(fallback_v, axis=0)
            if np.all(embedding != np.zeros(300)):
              print("{}'{}' mapped to the average of {}".format("  " * depth, word, fallback_w))
              break
          else:
              raise ValueError("Error fallback word is of type {}: {}".format(fallback_word, type(fallback_word)))
    if np.all(embedding == np.zeros(300)):
      print("{}Warning! Couldn't find semantic vector for '{}'".format("  " * depth, word))
      return embedding
    return embedding / np.linalg.norm(embedding)


# Get word embedding of subject and object label and concatenate them
def getSemanticVector(subject_label, object_label, emb_model):
    subject_vector = emb_model[subject_label]
    object_vector  = emb_model[object_label]
    return np.concatenate((subject_vector, object_vector), axis=0)


# data_info may be just the dataset name
def data_info_to_ds_args(data_info):
  if isinstance(data_info, str):
    data_info = {"name" : data_info}
  return data_info


# Wrapper for cv2.imread
def read_img(im_file):
  """ Wrapper for cv2.imread """
  if not os.path.exists(im_file):
    raise FileNotFoundError("Image file not found: " + im_file)
  return np.array(cv2.imread(im_file))

# LeveledAverageMeter, inspired from AverageMeter:
#  https://github.com/pytorch/examples/blob/490243127c02a5ea3348fa4981ecd7e9bcf6144c/imagenet/main.py#L359
# TODO: check that this works properly
class LeveledAverageMeter(object):
  """ Computes and stores the average and current value """
  def __init__(self, n_levels = 1):
    self._n_levels = n_levels

    self._cnt = [0] * self._n_levels
    self._val = [0] * self._n_levels
    self._sum = [0] * self._n_levels

  # Reset some layers' values
  def reset(self, level = 0):
    if level >= self._n_levels:
      raise ValueError("Error: LeveledAverageMeter can't reset more levels than it has")
    for i in range(level+1):
      self._cnt[i] = 0
      self._val[i] = 0
      self._sum[i] = 0

  # "Add" values to all layers
  def update(self, val, n = 1):
    for i in range(self._n_levels):
      self._cnt[i] += n
      self._val[i]  = val
      self._sum[i] += val * n

  def cnt(self, level = 0): return self._cnt[level]
  def val(self, level = 0): return self._val[level]
  def sum(self, level = 0): return self._sum[level]
  def avg(self, level = 0): return self.sum(level) / self.cnt(level)


# Smart frequency = a frequency that can be a relative (a precentage) or absolute (integer)
def smart_frequency_check(i_iter, num_iters, smart_frequency):
  if float(smart_frequency) == 0.0: return False
  if isinstance(smart_frequency, int):
    abs_freq = smart_frequency
  else:
    abs_freq = max(int(num_iters*smart_frequency),1)
  return (i_iter % abs_freq) == 0

def time_diff_str(time1, time2 = None):
  if time2 is None:
    dtime = time1
  else:
    dtime = time2 - time1
  return time.strftime('%H:%M:%S', time.gmtime(int(dtime)))

# https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-subobjects-chained-properties
from functools import reduce

def rsetattr(obj, attr, val):
  pre, _, post = attr.rpartition('.')
  return setattr(rgetattr(obj, pre) if pre else obj, post, val)

# using wonder's beautiful simplification: https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-objects/31174427?noredirect=1#comment86638618_31174427

def rgetattr(obj, attr, *args):
  def _getattr(obj, attr):
    return getattr(obj, attr, *args)
  return reduce(_getattr, [obj] + attr.split('.'))

# https://stackoverflow.com/a/52260663
def get(dataDict, mapList, dig = False):
  """Iterate nested dictionary"""
  if not dig:
    def dict_get(c,k):
      if c is None:
        return None
      if k in c and c[k] is None:
        c[k] = {}
      # print(("c,k,dict.get(c,k)",(c,k,dict.get(c,k))))
      return dict.get(c,k)

    # fun = dict.get
    fun = dict_get
  else:
    def dict_setget(c,k):
      if not k in c:
        c[k] = {}
      return dict.get(c,k)
    fun = dict_setget
  return reduce(fun, mapList, dataDict)

def listify(elem_or_list):
  return elem_or_list if isinstance(elem_or_list, list) else [elem_or_list]

def patch_key(d, old_key, new_key):
  old_key = listify(old_key)
  new_key = listify(new_key)
  if not get(d, old_key) is None:
    if not get(d, new_key) is None:
      raise ValueError("Patching dictionary failed: old = d[\"{}\"] = {},  new = d[\"{}\"] = {}".format(old_key, get(d, old_key), new_key, get(d, new_key)))
    else:
      last_new_key = new_key.pop()
      # print(("last_new_key", last_new_key))
      last_old_key = old_key.pop()
      # print(("last_old_key", last_old_key))
      new_d = get(d, new_key, True)
      # print(("new_d", new_d))
      old_d = get(d, old_key)
      # print(("old_d", old_d))
      new_d[last_new_key] = old_d.pop(last_old_key)
      # Deflate {}s
      while not old_d and len(old_key) > 0:
        last_key = old_key.pop()
        old_d = get(d, old_key)
        del old_d[last_key]
        # print((("last_key","old_d"),(last_key,old_d)))

# Read a list from a file, line-by-line
def load_txt_list(filename):
  l = []
  with  open(filename, 'r') as rfile:
    for index, line in enumerate(rfile):
      l.append(line.strip())
  return l

# Invert a dictionary or a list
def invert_dict(d):
  if isinstance(d, dict):   d_pairs = d.items()
  elif isinstance(d, list): d_pairs = enumerate(d)
  else: raise ValueError("Can't invert to dict object of type {}".format(repr(type(d))))
  return {v: k for k, v in d_pairs}
  # ? for list dict(zip(self.d, xrange(self.d)))

# Config File

def cfg_from_file(filename):
  """Load a config file and merge it into the default options."""
  import yaml
  try:
    from yaml import CLoader as Loader, CDumper as Dumper
  except ImportError:
    from yaml import Loader, Dumper

  with open(filename, 'r') as f:
    yaml_cfg = (yaml.load(f, Loader = Loader))

  return yaml_cfg

def dict_patch(a, def_dict):
  """ Patch a dictionary of default values def_dict with a dictionary a, clobbering the
  options in def_dict whenever they are also specified in a. """
  def_dict = deepcopy(def_dict)

  for k, v in a.items():
    #print(k,v)
    # a must specify keys that are in def already (although this may not be desirable in some cases)
    if k not in def_dict:
      warnings.warn("dict_patch: key '{}' not int original dict.".format(k), UserWarning)
      # raise KeyError("{} is not a valid config key".format(k))

    # Type check
    # if not isinstance(v, type(def_dict[k])):
    #   if isinstance(def_dict[k], np.ndarray):
    #     v = np.array(v, dtype=def_dict[k].dtype)
    #   else:
    #     raise ValueError(("Type mismatch ({} vs. {}) "
    #                       "for config key: {}").format(type(def_dict[k]),
    #                                                    type(v), k))

    # Recursively merge dicts
    #print(v, isinstance(v,dict))
    if isinstance(v,dict) or isinstance(v,munch.Munch):
      try:    def_dict[k] = dict_patch(a[k], def_dict[k])
      except: raise ValueError("Error under config key: {}".format(k))
    else:
      #print(def_dict, k, v)
      def_dict[k] = v
      #print(def_dict)
      #print()
  #print(def_dict)
  #print()
  return def_dict

"""
import importlib
importlib.reload(utils)
a = {'1': 'ciao', '2': {'21': {'YEAH':{'YEAH':{'YEAH':"yeah"}}}, '22': 'YEAH'}, '3': 'ciaos'}
utils.patch_key(a, ['2', '21', 'YEAH', 'YEAH', 'YEAH'], '4')
a
utils.patch_key(a, '4', ['2', '21', 'YEAH', 'YEAH'])
a
# utils.patch_key(a, ['2', '22'], '4')
# a
# utils.patch_key(a, '4', ['2', '22'])
# a
"""
