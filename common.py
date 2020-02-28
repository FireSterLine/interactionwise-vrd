from os.path import join

images_dir = "images"
images_det_dir = "images_det"

faster_rcnn_dir = "faster-rcnn"
faster_rcnn_models_dir = join(faster_rcnn_dir, "models")


import os
import cv2
import numpy as np

# Wrapper for cv2.imread
def read_img(im_file):
  if not os.path.exists(im_file):
    raise Exception("Image file not found: " + im_file)
  return np.array(cv2.imread(im_file))
