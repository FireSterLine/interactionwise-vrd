import os
import cv2
import numpy as np

# TODO: figure out what pixel means to use, how to compute them:
#  do they come from the dataset used for training, perhaps?
vrd_pixel_means = np.array([[[102.9801, 115.9465, 122.7717]]])


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

# Wrapper for cv2.imread
def read_img(im_file):
  if not os.path.exists(im_file):
    raise Exception("Image file not found: " + im_file)
  return np.array(cv2.imread(im_file))
