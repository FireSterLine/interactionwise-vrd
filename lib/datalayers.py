import numpy as np
import os.path as osp
import scipy.io as sio
import scipy
import cv2
import pickle
import sys
from lib.blob import prep_im_for_blob
from lib.dataset
import math

# TODO: expand so that it supports batch sizes > 1
class VRDDataLayer(object):

    def __init__(self, ds_info, stage):
        """ Setup the VRD DataLayer """

        if isinstance(ds_info, str):
            ds_name = ds_info
            ds_args = {}
        else:
            ds_name = ds_info["ds_name"]
            del ds_info["ds_name"]
            ds_args = ds_info

        self.ds_name = ds_name
        self.stage   = stage

        self.dataset = dataset(self.ds_name, **ds_args)

        self.n_obj   = self.dataset.n_obj
        self.n_pred  = self.dataset.n_pred

        self.anno = self.dataset.getAnno()
        self.n_anno = len(self.anno)

        self.cur_anno = 0

    def step(self):
        """Get blobs and copy them into this layer's top blob vector."""
        anno_img = self.anno[self.cur]
        im_path = anno_img['img_path']
        im = cv2.imread(im_path)
        ih = im.shape[0]
        iw = im.shape[1]
        PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])
        image_blob, im_scale = prep_im_for_blob(im, PIXEL_MEANS)
        blob = np.zeros((1,)+image_blob.shape, dtype=np.float32)
        blob[0] = image_blob
        boxes = np.zeros((anno_img['boxes'].shape[0], 5))
        boxes[:, 1:5] = anno_img['boxes'] * im_scale
        classes = np.array(anno_img['classes'])
        ix1 = np.array(anno_img['ix1'])
        ix2 = np.array(anno_img['ix2'])
        rel_classes = anno_img['rel_classes']

        n_rel_inst = len(rel_classes)
        rel_boxes = np.zeros((n_rel_inst, 5))
        rel_labels = -1*np.ones((1, n_rel_inst*self._num_relations))
        SpatialFea = np.zeros((n_rel_inst, 2, 32, 32))
        rel_so_prior = np.zeros((n_rel_inst, self._num_relations))
        pos_idx = 0
        for ii in range(len(rel_classes)):
            sBBox = anno_img['boxes'][ix1[ii]]
            oBBox = anno_img['boxes'][ix2[ii]]
            rBBox = utils.getUnionBBox(sBBox, oBBox, ih, iw)
            rel_boxes[ii, 1:5] = np.array(rBBox) * im_scale
            SpatialFea[ii] = [self._getDualMask(ih, iw, sBBox), \
                              self._getDualMask(ih, iw, oBBox)]
            rel_so_prior[ii] = self._so_prior[classes[ix1[ii]], classes[ix2[ii]]]
            for r in rel_classes[ii]:
                rel_labels[0, pos_idx] = ii*self._num_relations + r
                pos_idx += 1
        image_blob = image_blob.astype(np.float32, copy=False)
        boxes = boxes.astype(np.float32, copy=False)
        classes = classes.astype(np.float32, copy=False)
        self._cur += 1
        if(self._cur >= len(self._anno)):
            self._cur = 0
        return blob, boxes, rel_boxes, SpatialFea, classes, ix1, ix2, rel_labels, rel_so_prior

if __name__ == '__main__':
    pass
