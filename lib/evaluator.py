import numpy as np
import torch

from lib.datalayers import VRDDataLayer
from lib.evaluation_dsr import eval_recall_at_N, eval_obj_img # TODO remove this module
import time

# TODO: remove this and use dataset.dir instead
import globals

# TODO: check, is all of this using the GPU, or can we improve the time?
class VRDEvaluator():
  """ Evaluator for Predicate Prediction and Relationship Prediction """

  def __init__(self, data_args, args = { "use_obj_prior" : True }):
    self.data_args = data_args
    self.args = args

  def test_pre(self, vrd_model):
    """ Test model on Predicate Prediction """
    with torch.no_grad():
      vrd_model.eval()
      time1 = time.time()

      ...
      # TODO: just one VRD test layer
      test_data_layer = VRDDataLayer(self.data_args, "test")

      rlp_labels_cell  = []
      tuple_confs_cell = []
      sub_bboxes_cell  = []
      obj_bboxes_cell  = []

      N = 100 # What's this? (num of rel_res) (with this you can compute R@i for any i<=N)

      while True:

        try:
          net_input, obj_classes_out, ori_bboxes = test_data_layer.next()
        except StopIteration:
          print("StopIteration")
          break

        if(net_input is None):
          rlp_labels_cell.append(None)
          tuple_confs_cell.append(None)
          sub_bboxes_cell.append(None)
          obj_bboxes_cell.append(None)
          continue

        img_blob, obj_boxes, u_boxes, idx_s, idx_o, spatial_features, obj_classes = net_input

        tuple_confs_im = np.zeros((N,),   dtype = np.float) # Confidence...
        rlp_labels_im  = np.zeros((N, 3), dtype = np.float) # Rel triples
        sub_bboxes_im  = np.zeros((N, 4), dtype = np.float) # Subj bboxes
        obj_bboxes_im  = np.zeros((N, 4), dtype = np.float) # Obj bboxes

        obj_scores, rel_scores = vrd_model(*net_input)
        rel_prob = rel_scores.data.cpu().numpy()
        rel_res = np.dstack(np.unravel_index(np.argsort(-rel_prob.ravel()), rel_prob.shape))[0][:N]

        for ii in range(rel_res.shape[0]):
          rel = rel_res[ii, 1]
          tuple_idx = rel_res[ii, 0]

          conf = rel_prob[tuple_idx, rel]
          tuple_confs_im[ii] = conf

          rlp_labels_im[ii] = [obj_classes_out[idx_s[tuple_idx]], rel, obj_classes_out[idx_o[tuple_idx]]]

          sub_bboxes_im[ii] = ori_bboxes[idx_s[tuple_idx]]
          obj_bboxes_im[ii] = ori_bboxes[idx_o[tuple_idx]]

        # TODO: check
        # Is this because of the background ... ? If so, use proper flags instead of the name...
        if(self.data_args.name == "vrd"):
          rlp_labels_im += 1

        tuple_confs_cell.append(tuple_confs_im)
        rlp_labels_cell.append(rlp_labels_im)
        sub_bboxes_cell.append(sub_bboxes_im)
        obj_bboxes_cell.append(obj_bboxes_im)

      res = {
        "rlp_confs_ours"  : tuple_confs_cell,
        "rlp_labels_ours" : rlp_labels_cell,
        "sub_bboxes_ours" : sub_bboxes_cell,
        "obj_bboxes_ours" : obj_bboxes_cell,
      }

      rec_50     = eval_recall_at_N(self.data_args.name, 50,  res, use_zero_shot = False)
      rec_50_zs  = eval_recall_at_N(self.data_args.name, 50,  res, use_zero_shot = True)
      rec_100    = eval_recall_at_N(self.data_args.name, 100, res, use_zero_shot = False)
      rec_100_zs = eval_recall_at_N(self.data_args.name, 100, res, use_zero_shot = True)
      time2 = time.time()

      return rec_50, rec_50_zs, rec_100, rec_100_zs, (time2-time1)

  # Relationship Prediction
  def test_rel(self, vrd_model):
    """ Test model on Relationship Prediction """
    with torch.no_grad():
      vrd_model.eval()
      time1 = time.time()

      test_data_layer = VRDDataLayer(self.data_args.name, "test")

      with open(osp.join(globals.data_dir, self.data_args.name, "test.pkl"), 'rb') as fid:
        anno = pickle.load(fid, encoding="latin1")

      # TODO: proposals is not ordered, but a dictionary with im_path keys
      # TODO: expand so that we don't need the proposals pickle, and we generate it if it's not there, using Faster-RCNN?
      # TODO: move the proposals file path to a different one (maybe in Faster-RCNN)
      with open(osp.join(globals.data_dir, self.data_args.name, "eval", "det_res.pkl"), 'rb') as fid:
        proposals = pickle.load(fid)
        # TODO: zip these
        pred_boxes   = proposals["boxes"]
        pred_classes = proposals["cls"]
        pred_confs   = proposals["confs"]

      N = 100 # What's this? (num of rel_res) (with this you can compute R@i for any i<=N)

      pos_num = 0.0
      loc_num = 0.0
      gt_num  = 0.0

      rlp_labels_cell  = []
      tuple_confs_cell = []
      sub_bboxes_cell  = []
      obj_bboxes_cell  = []
      predict = []

      if len(anno) != len(proposals["cls"]):
        print("ERROR: something is wrong in prediction: {} != {}".format(len(anno), len(proposals["cls"])))

      for step,anno_img in enumerate(anno):

        objdet_res = {
          "boxes"   : pred_boxes[step],
          "classes" : pred_classes[step].reshape(-1),
          "confs"   : pred_confs[step].reshape(-1)
        }

        try:
          net_input, obj_classes_out, rel_sop_prior, ori_bboxes = test_data_layer.next(objdet_res)
        except StopIteration:
          print("StopIteration")
          break

        if(net_input is None):
          rlp_labels_cell.append(None)
          tuple_confs_cell.append(None)
          sub_bboxes_cell.append(None)
          obj_bboxes_cell.append(None)
          continue

        gt_boxes = anno_img["boxes"].astype(np.float32)
        gt_cls = np.array(anno_img["classes"]).astype(np.float32)

        obj_score, rel_score = vrd_model(*net_input) # img_blob, obj_boxes, u_boxes, idx_s, idx_o, spatial_features, obj_classes)

        _, obj_pred = obj_score[:, 1::].data.topk(1, 1, True, True)
        obj_score = F.softmax(obj_score, dim=1)[:, 1::].data.cpu().numpy()

        rel_prob = rel_score.data.cpu().numpy()
        rel_prob += np.log(0.5*(rel_sop_prior+1.0 / test_data_layer.n_pred))

        img_blob, obj_boxes, u_boxes, idx_s, idx_o, spatial_features, obj_classes = net_input

        pos_num_img, loc_num_img = eval_obj_img(gt_boxes, gt_cls, ori_bboxes, obj_pred.cpu().numpy(), gt_thr=0.5)
        pos_num += pos_num_img
        loc_num += loc_num_img
        gt_num  += gt_boxes.shape[0]

        tuple_confs_im = []
        rlp_labels_im  = np.zeros((rel_prob.shape[0]*rel_prob.shape[1], 3), dtype = np.float)
        sub_bboxes_im  = np.zeros((rel_prob.shape[0]*rel_prob.shape[1], 4), dtype = np.float)
        obj_bboxes_im  = np.zeros((rel_prob.shape[0]*rel_prob.shape[1], 4), dtype = np.float)
        n_idx = 0

        for tuple_idx in range(rel_prob.shape[0]):
          for rel in range(rel_prob.shape[1]):
            if(self.args.use_obj_prior):
              if(objdet_res["confs"].ndim == 1):
                conf = np.log(objdet_res["confs"][idx_s[tuple_idx]]) + np.log(objdet_res["confs"][idx_o[tuple_idx]]) + rel_prob[tuple_idx, rel]
              else:
                conf = np.log(objdet_res["confs"][idx_s[tuple_idx], 0]) + np.log(objdet_res["confs"][idx_o[tuple_idx], 0]) + rel_prob[tuple_idx, rel]
            else:
              conf = rel_prob[tuple_idx, rel]
            tuple_confs_im.append(conf)
            sub_bboxes_im[n_idx] = ori_bboxes[idx_s[tuple_idx]]
            obj_bboxes_im[n_idx] = ori_bboxes[idx_o[tuple_idx]]
            rlp_labels_im[n_idx] = [obj_classes_out[idx_s[tuple_idx]], rel, obj_classes_out[idx_o[tuple_idx]]]
            n_idx += 1

        # TODO: check
        # Is this because of the background ... ?
        if(self.data_args.name == "vrd"):
          rlp_labels_im += 1

        # Why is this needed? ...
        tuple_confs_im = np.array(tuple_confs_im)
        idx_order = tuple_confs_im.argsort()[::-1][:N]
        rlp_labels_im = rlp_labels_im[idx_order,:]
        tuple_confs_im = tuple_confs_im[idx_order]
        sub_bboxes_im  = sub_bboxes_im[idx_order,:]
        obj_bboxes_im  = obj_bboxes_im[idx_order,:]

        rlp_labels_cell.append(rlp_labels_im)
        tuple_confs_cell.append(tuple_confs_im)
        sub_bboxes_cell.append(sub_bboxes_im)
        obj_bboxes_cell.append(obj_bboxes_im)

        step += 1

      res = {
        "rlp_confs_ours"  : tuple_confs_cell,
        "rlp_labels_ours" : rlp_labels_cell,
        "sub_bboxes_ours" : sub_bboxes_cell,
        "obj_bboxes_ours" : obj_bboxes_cell,
      }

      if len(anno) != len(res["obj_bboxes_ours"]):
        print("ERROR: something is wrong in prediction: {} != {}".format(len(anno), len(res["obj_bboxes_ours"])))

      rec_50     = eval_recall_at_N(self.data_args.name, 50,  res, use_zero_shot = False)
      rec_50_zs  = eval_recall_at_N(self.data_args.name, 50,  res, use_zero_shot = True)
      rec_100    = eval_recall_at_N(self.data_args.name, 100, res, use_zero_shot = False)
      rec_100_zs = eval_recall_at_N(self.data_args.name, 100, res, use_zero_shot = True)
      time2 = time.time()

      return rec_50, rec_50_zs, rec_100, rec_100_zs, pos_num, loc_num, gt_num, (time2 - time1)
