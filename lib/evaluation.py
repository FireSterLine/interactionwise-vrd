# TODO: don't use this, make our own
# Is it possible to use the matlab code from vrd?
#  https://github.com/Prof-Lu-Cewu/Visual-Relationship-Detection/tree/master/evaluation

import scipy.io as sio
import numpy as np
import pickle
import copy
import time
import sys
import pdb
import os.path as osp
this_dir = osp.dirname(osp.realpath(__file__))
# print this_dir

def eval_per_image(i, gt, pred, use_rel, gt_thr = 0.5, return_match = False):
  gt_tupLabel = gt["tuple_label"][i].astype(np.float32)
  num_gt_tuple = gt_tupLabel.shape[0]
  #print("gt_tuples: ")
  #for tup in gt_tupLabel:
  #  print(tup)
  #print("pred_tuples: ")
  #for tup in pred["tuple_label"]:
  #  print(tup)
  #print(gt_tupLabel.shape)
  if num_gt_tuple == 0 or pred["tuple_confs"][i] is None:
    #print(num_gt_tuple, len(pred["tuple_confs"][i]))
    return 0, 0
  if not use_rel:
    gt_tupLabel = gt_tupLabel[:, (0,2)]
  gt_objBox = gt["obj_bboxes"][i].astype(np.float32)
  gt_subBox = gt["sub_bboxes"][i].astype(np.float32)

  gt_detected = np.zeros((num_gt_tuple, 1), np.float32)
  labels = pred["tuple_label"][i].astype(np.float32)
  boxSub = pred["sub_bboxes"][i].astype(np.float32)
  boxObj = pred["obj_bboxes"][i].astype(np.float32)
  num_tuple = labels.shape[0]

  tp = np.zeros((num_tuple,1))
  fp = np.zeros((num_tuple,1))
  for j in range(num_tuple):
    bbO = boxObj[j,:]
    bbS = boxSub[j,:]
    ovmax = gt_thr
    kmax = -1
    for k in range(num_gt_tuple):
      if np.linalg.norm(labels[j,:] - gt_tupLabel[k,:],2) != 0:
        continue;
      if gt_detected[k] > 0:
        continue;
      # [xmin, ymin, xmax, ymax]
      bbgtO = gt_objBox[k,:];
      bbgtS = gt_subBox[k,:];

      biO = np.array([max(bbO[0],bbgtO[0]), max(bbO[1],bbgtO[1]), min(bbO[2],bbgtO[2]), min(bbO[3],bbgtO[3])])
      iwO=biO[2]-biO[0]+1;
      ihO=biO[3]-biO[1]+1;

      biS = np.array([max(bbS[0],bbgtS[0]), max(bbS[1],bbgtS[1]), min(bbS[2],bbgtS[2]), min(bbS[3],bbgtS[3])])
      iwS=biS[2]-biS[0]+1;
      ihS=biS[3]-biS[1]+1;

      if iwO>0 and ihO>0 and iwS>0 and ihS>0:
        # compute overlap as area of intersection / area of union
        uaO=(bbO[2]-bbO[0]+1)*(bbO[3]-bbO[1]+1) + \
            (bbgtO[2]-bbgtO[0]+1)*(bbgtO[3]-bbgtO[1]+1) - iwO*ihO;
        ovO =iwO*ihO/uaO;

        uaS=(bbS[2]-bbS[0]+1)*(bbS[3]-bbS[1]+1) + \
            (bbgtS[2]-bbgtS[0]+1)*(bbgtS[3]-bbgtS[1]+1) - iwS*ihS
        ovS =iwS*ihS/uaS
        ov = min(ovO, ovS)

        # makes sure that this object is detected according
        #   to its individual threshold
        if ov >= ovmax:
          ovmax=ov;
          kmax=k;
    if kmax > -1:
      tp[j] = 1;
      gt_detected[kmax] = 1;
    else:
      fp[j] = 1;
  if return_match:
    return tp
  return tp.sum(), num_gt_tuple

# Recall quantifies the number of positive class predictions
#   made out of all positive examples in the dataset.
def eval_recall_at_N(res, gts, Ns = [100, 50, 4.], num_imgs = None, use_rel = True):

  # If not specified, num_imgs is the length of the ground truth
  valid_gts = [gt for gt in gts if gt is not None]
  if len(valid_gts) == 0:
    print("Warning! No valid ground-truths were provided. Can't really evaluate")

  for i_gt in range(len(valid_gts)-1):
    if len(valid_gts[i_gt]["obj_bboxes"]) != len(valid_gts[i_gt+1]["obj_bboxes"]):
      print("Warning! Ground truths provided do not share the same length: test performance might be off! {} != {}".format(len(valid_gts[i_gt]["obj_bboxes"]), len(valid_gts[i_gt+1]["obj_bboxes"])))

  if num_imgs is None and len(valid_gts):
    num_imgs = len(valid_gts[0]["obj_bboxes"])

  test_set_size = min(num_imgs, len(res["rlp_confs_ours"]))
  if num_imgs != len(res["rlp_confs_ours"]):
    print("Warning! Test results and ground truths do not have the same length: test performance might be off! {} != {}. The minimum will be used: {}".format(num_imgs, len(res["rlp_confs_ours"]), test_set_size))

  Ns = sorted(Ns, reverse=True)
  max_N = Ns[0]
  there_are_float_Ns = len([1 for N in Ns if isinstance(N, float)]) > 0

  base_pred = {}
  base_pred["tuple_label"] = copy.deepcopy(res["rlp_labels_ours"])
  base_pred["tuple_confs"] = copy.deepcopy(res["rlp_confs_ours"])
  base_pred["sub_bboxes"]  = copy.deepcopy(res["sub_bboxes_ours"])
  base_pred["obj_bboxes"]  = copy.deepcopy(res["obj_bboxes_ours"])

  # Sort by confidence
  for i,(tuple_labels, tuple_confs, sub_bboxes, obj_bboxes) in enumerate(zip(base_pred["tuple_label"], base_pred["tuple_confs"], base_pred["sub_bboxes"], base_pred["obj_bboxes"])):
    if i > test_set_size: break
    if(tuple_confs is None): continue
    tuple_confs = np.array(tuple_confs)
    if(tuple_confs.shape[0] == 0): continue
    if not there_are_float_Ns:
      idx_order = tuple_confs.argsort()[::-1][:max_N]
    else:
      idx_order = tuple_confs.argsort()[::-1]
    base_pred["tuple_label"][i]  = tuple_labels[idx_order,:]
    base_pred["tuple_confs"][i]  = tuple_confs[idx_order]
    base_pred["sub_bboxes"][i]   = sub_bboxes[idx_order,:]
    base_pred["obj_bboxes"][i]   = obj_bboxes[idx_order,:]

    if len(idx_order) < max_N:
      raise ValueError("Can't compute R@{}: input is malformed (idx_order.shape: {}, pred[\"tuple_confs\"][{}].shape: {}".format(max_N, idx_order.shape, ii, base_pred["tuple_confs"][ii].shape))

  def get_recall(pred, this_gt):
    # Evaluate each image
    tp_num = 0
    num_pos_tuple = 0
    for i in range(test_set_size):
      img_tp, img_gt = eval_per_image(i, this_gt, pred, use_rel, gt_thr = 0.5)
      tp_num += img_tp
      num_pos_tuple += img_gt
    return (np.float64(tp_num)/num_pos_tuple)*100

  recalls = []

  for N in Ns:
    if there_are_float_Ns: # float_Ns break the "pyramidality" of R@x and R@y for x > y
      pred = copy.deepcopy(base_pred)
    else:
      pred = base_pred

    if not isinstance(N, float):
      for i,(tuple_labels, tuple_confs, sub_bboxes, obj_bboxes) in enumerate(zip(pred["tuple_label"], pred["tuple_confs"], pred["sub_bboxes"], pred["obj_bboxes"])):
        if i > test_set_size: break
        if(tuple_confs is None): continue
        pred["tuple_label"][i]  = pred["tuple_label"][i][:N]
        pred["tuple_confs"][i]  = pred["tuple_confs"][i][:N]
        pred["sub_bboxes"][i]   = pred["sub_bboxes"][i][:N]
        pred["obj_bboxes"][i]   = pred["obj_bboxes"][i][:N]
    for gt in gts:
      if gt is None:
        recalls.append(np.nan)
        continue
      if isinstance(N, float): # TODO: validate
        for i,(tuple_labels, tuple_confs, sub_bboxes, obj_bboxes) in enumerate(zip(pred["tuple_label"], pred["tuple_confs"], pred["sub_bboxes"], pred["obj_bboxes"])):
          if i > test_set_size: break
          if(tuple_confs is None): continue
          x = int(np.ceil(N * len(gt["tuple_label"][i])))
          pred["tuple_label"][i]  = pred["tuple_label"][i][:x]
          pred["tuple_confs"][i]  = pred["tuple_confs"][i][:x]
          pred["sub_bboxes"][i]   = pred["sub_bboxes"][i][:x]
          pred["obj_bboxes"][i]   = pred["obj_bboxes"][i][:x]
          # print(x)
      recalls.append(get_recall(pred, gt))

  return tuple(recalls)

def eval_obj_img(gt_boxes, gt_cls, pred_boxes, pred_cls, gt_thr=0.5, return_flag = False):
  pos_num = 0;
  loc_num = 0;
  dets = []
  gts = []
  for ii in range(pred_boxes.shape[0]):
    recog_flag = -1;
    gt_flag = -1;
    bbox = pred_boxes[ii]
    for jj in range(gt_boxes.shape[0]):
      gt_box = gt_boxes[jj]

      in_box = np.array([max(bbox[0],gt_box[0]), max(bbox[1],gt_box[1]), min(bbox[2],gt_box[2]), min(bbox[3],gt_box[3])])
      in_box_w = in_box[2]-in_box[0]+1;
      in_box_h = in_box[3]-in_box[1]+1;

      if in_box_w>0 and in_box_h>0:
        # compute overlap as area of intersection / area of union
        un_box_area = 1.0*(bbox[2]-bbox[0]+1)*(bbox[3]-bbox[1]+1) + \
                     (gt_box[2]-gt_box[0]+1)*(gt_box[3]-gt_box[1]+1) - in_box_w*in_box_h;
        IoU = 1.0*in_box_w*in_box_h/un_box_area;
        # makes sure that this object is detected according
        # to its individual threshold
        if IoU >= gt_thr:
          gt_flag = gt_cls[jj]
          if np.linalg.norm(gt_cls[jj] - pred_cls[ii]) == 0:
            recog_flag = 1
            break
          else:
            recog_flag = 2
    if(recog_flag == 1):
      pos_num += 1
    elif(recog_flag == 2):
      loc_num += 1
    dets.append(recog_flag)
    gts.append(gt_flag)
  if(return_flag):
    return dets, gts
  return pos_num, loc_num

# What to do with this?
def eval_object_recognition_top_N(proposals_path):
  raise NotImplementedError
  with open("data/vrd/test.pkl", 'rb') as fid:
    anno = pickle.load(fid)

  with open(proposals_path, 'rb') as fid:
    proposals = pickle.load(fid)

  pos_num = 0.0
  loc_num = 0.0
  for ii in range(len(anno)):
    if(anno[ii] is None):
      continue
    anno_img = anno[ii]
    gt_boxes = anno_img["boxes"].astype(np.float32)
    gt_cls = anno_img["classes"].astype(np.float32)
    pred_boxes = proposals["boxes"][ii].astype(np.float32)
    pred_cls = proposals["cls"][ii].astype(np.float32)
    pos_num_img, loc_num_img = eval_obj_img(gt_boxes, gt_cls, pred_boxes, pred_cls)
    pos_num += pos_num_img
    loc_num += loc_num_img
  print(pos_num/(pos_num+loc_num))

if __name__ == "__main__":
  pass
