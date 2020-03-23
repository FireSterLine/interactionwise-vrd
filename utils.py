import os
import cv2
import numpy as np
import model.utils.net_utils as frcnn_net_utils
import time
from copy import deepcopy
import torch

weights_normal_init  = frcnn_net_utils.weights_normal_init
save_checkpoint      = frcnn_net_utils.save_checkpoint
adjust_learning_rate = frcnn_net_utils.adjust_learning_rate

# TODO: figure out what pixel means to use, how to compute them:
#  do they come from the dataset used for training, perhaps?
# Read here: https://github.com/GriffinLiang/vrd-dsr/issues/12
# If you use the pretrained, you should use the same value. Boh
vrd_pixel_means = np.array([[[102.9801, 115.9465, 122.7717]]])

# Pytorch CUDA Fallback
# TODO: check if this works and then use it everywhere instead of cuda()
try:
  device = torch.device("cuda")
  torch.LongTensor(device=device)
except RuntimeError:
  device = torch.device("cpu")
  torch.LongTensor(device=device)


# Bbox as a dict to numpy array
def bboxDictToNumpy(bbox_dict):
  return np.array([bbox_dict["xmin"],
                    bbox_dict["ymin"],
                    bbox_dict["xmax"],
                    bbox_dict["ymax"]])


# Union box of two boxes
def getUnionBBox(aBB, bBB, ih, iw, margin=10):
  return [max(0, min(aBB[0], bBB[0]) - margin),
          max(0, min(aBB[1], bBB[1]) - margin),
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

# Get word embedding of subject and object label and concatenate them
def getSemanticVector(subject_label, object_label, w2v_model):
  # the key errors mean that the word was not found in the model's dictionary
  try:
    subject_vector = w2v_model[subject_label]
  except KeyError:
    subject_vector = np.zeros(300)

  try:
    object_vector = w2v_model[object_label]
  except KeyError:
    object_vector = np.zeros(300)
  combined_vector = np.concatenate((subject_vector, object_vector), axis=0)
  return combined_vector

# data_info may be just the dataset name
def data_info_to_data_args(data_info):
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
  def avg(self, level = 0): return self.sum(level) / self.count(level)


# Smart frequency = a frequency that can be a relative (a precentage) or absolute (integer)
def smart_fequency_check(iter, num_iters, smart_frequency):
  if isinstance(smart_frequency, int):
    abs_freq = smart_frequency
  else:
    abs_freq = int(num_iters*smart_frequency)
  return (iter % abs_freq) == 0

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
    def dict_setdef(c,k):
      c[k] = c.get(k, {})
      return c[k]

    if not dig:
      fun = dict.get
    else:
      fun = dict_setdef

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
      last_old_key = old_key.pop()
      new_d = get(d, new_key, True)
      old_d = get(d, old_key)
      d[last_new_key] = old_d.pop(last_old_key)

a ={'1': 'ciao', '2': {'21': 'YEAH', '22': 'YEAH'}, '3': 'ciaos'}
patch_key(a, ['2', '21'], '4')
a
