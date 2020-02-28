from os.path import join

images_dir = "images"
images_det_dir = "images_det"
models_dir = "models"

faster_rcnn_dir = "faster-rcnn"
faster_rcnn_models_dir = join(faster_rcnn_dir, "models")

import os
import cv2
import numpy as np


# TODO: figure out what pixel means to use, how to compute them:
#  do they come from the dataset used for training, perhaps?
vrd_pixel_means = np.array([[[102.9801, 115.9465, 122.7717]]])


# Wrapper for cv2.imread
def read_img(im_file):
  if not os.path.exists(im_file):
    raise Exception("Image file not found: " + im_file)
  return np.array(cv2.imread(im_file))
