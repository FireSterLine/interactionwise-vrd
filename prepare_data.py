import os
import shutil
import sys
import os.path as osp
from glob import glob

from collections import defaultdict

import numpy as np
import scipy.io as sio

from gensim.models import KeyedVectors
from gensim.models import Word2Vec
from gensim.test.utils import datapath, get_tmpfile
from gensim.scripts.glove2word2vec import glove2word2vec

import json
import pickle
import globals
import utils
import random
from copy import deepcopy

from data.genome.clean_vg import VGCleaner
from scripts.train_word2vec import VRDEmbedding, EpochLogger, EpochSaver
from glove import Glove

# Prepare the data without saving anything at all
DRY_RUN = False

# Re-sort predicate categories by # of occurrences in the train set
RESORT_PREDICATES_BY_TRAIN_COUNTS = True

"""
This script prepares the data from the input format to the one we want.
Each dataset has potentially a different format, so we use an abstract class DataPreparer
and introduce a different class for handling each input format.

Notes:
- The granularity (either 'img' or 'rel') of an output format refers to whether
   the relationships are grouped by image or not.
  When granularity='rel', each relationship within an image will be a separate instance, as
   opposed to a set of relationships within an image instance. This is so to facilitate proper
   batching via the PyTorch Dataloader
- An image "annos" is something like {"objects" : [obj1, obj2, ...], "relationships" : [rel1, rel2, ...]}
- An image "relst" is something like [rel1, rel2, ...]
- When a database is loaded with load_data, it can be loaded in any convenient format and the flag
   self.cur_dformat will be set accordingly
- Then, one can switch from one format to the other with databse-independent functions.
  This makes sure that the data is the same across different formats

"""

class DataPreparer:
    def __init__(self, multi_label = True, generate_emb = []):

        self.multi_label  = multi_label
        self.generate_emb = generate_emb

        self.obj_vocab    = None
        self.pred_vocab   = None

        self.dir         = None
        self.vrd_data    = None
        self.cur_dformat = None
        self.splits      = None
        self.prepare_obj_det_fun = None

        # This flag tracks if something has been written already. Used for safety when handling data
        self.already_wrote_something = False
        self.pred_resorted = False
        # TODO move soP computations from dataset.py to prepare_data.py
        self.to_delete = [] #["soP.pkl", "pP.pkl"]
        #self.to_delete = ["soP.pkl", "pP.pkl"]

    # This function reads the dataset's vocab txt files and loads them
    def prepare_vocabs(self, obj_vocab_file, pred_vocab_file, subset = False):
        print("\tPreparing vocabs...")
        obj_vocab  = utils.load_txt_list(self.fullpath(obj_vocab_file))
        pred_vocab = utils.load_txt_list(self.fullpath(pred_vocab_file))

        if subset is not False and subset != "all":
          subset_mapping = self.readjson("subset_{}.json".format(subset))
          def rename_vocab(cls_vocab, rename_map):
            if rename_map is None: return
            for cls in range(len(cls_vocab)):
              if cls_vocab[cls] in rename_map:
                cls_vocab[cls] = rename_map[cls_vocab[cls]]
          def cleaned_vocab(cls_vocab, cl_map, cl_subset):
            new_vocab = []
            cls_map = {}
            # MAP
            for i_old_cls,cls in enumerate(cls_vocab):
              if (len(cl_subset) == 0 or cls in cl_subset) and not cls in cl_map:
                new_vocab.append(cls)
                cls_map[i_old_cls] = len(new_vocab)-1
            for cls,to_cls in cl_map.items():
              if (len(cl_subset) == 0 or to_cls in cl_subset):
                cls_map[cls_vocab.index(cls)] = new_vocab.index(to_cls)
            return new_vocab, cls_map
          # assert subset_mapping["obj_map"] == {}, "NotImplemented: A cleaning map for objects requires some preprocessing to the object_detections as well, if not a re-train of the object detection model. Better not touch these things."
          # obj_vocab,  obj_cls_map  = cleaned_vocab(obj_vocab,  subset_mapping["obj_map"])

          rename_vocab(pred_vocab, subset_mapping.get("pred_rename",None))
          pred_vocab, pred_cls_map = cleaned_vocab(pred_vocab, subset_mapping["pred_map"], subset_mapping["pred_subset"])
          self.subset_mapping = {"pred" : pred_cls_map}

        self.obj_vocab  = obj_vocab
        self.pred_vocab = pred_vocab

    # These functions are used for each different "dataset source" to convert the annotations to our formats
    def _generate_relst(self, anns): raise NotImplementedError
    def _generate_annos(self, anns): raise NotImplementedError
    
    def get_clean_obj_cls(self,  cls): # TODO: validate this
      return self.subset_mapping["obj"].get(cls, None) if (hasattr(self, "subset_mapping") and "obj" in self.subset_mapping) else cls
    def get_clean_pred_cls(self, cls):
      if self.pred_resorted: print("ERROR! Can't read more data when predicates have been resorted")
      #print(self.subset_mapping["pred"])
      #print(cls)
      #print(self.subset_mapping["pred"][cls])
      return self.subset_mapping["pred"].get(cls, None) if (hasattr(self, "subset_mapping") and "pred" in self.subset_mapping) else cls


    def _read_data_check(self):
      if self.pred_resorted:
        print("ERROR! Can't read more data when predicates have been resorted")
        input()
        exit(0)


    # Save data according to the specified formats
    def save_data(self, dformats):

      self.save_counts()
      self.prepareGT()
      if not self.prepare_obj_det_fun is None:
        self.prepare_obj_det_fun()

      # Remove existing files before saving
      for to_delete in self.to_delete:
        f = self.fullpath(to_delete, outputdir = True)
        if os.path.exists(f):
          os.remove(f)

      # Save vocabs
      print("\tGenerating vocabs...")
      self.writejson(self.obj_vocab,  "objects.json")
      self.writejson(self.pred_vocab, "predicates.json")

      # Save embeddings
      print("\tGenerating embeddings...")
      for model_name in self.generate_emb:
        model = load_emb_model(model_name)
        obj_emb  = [ getWordEmbedding(obj_label,  model, model_name).astype(float).tolist() for  obj_label in self.obj_vocab]
        pred_emb = [ getWordEmbedding(pred_label, model, model_name).astype(float).tolist() for pred_label in self.pred_vocab]
        self.writejson(obj_emb,  "objects-emb-{}.json".format(model_name))
        self.writejson(pred_emb, "predicates-emb-{}.json".format(model_name))

      # Save actual data
      for dformat in dformats:
        print("\tGenerating '{}' data...".format(dformat))
        granularity = "img"
        if isinstance(dformat, tuple):
          dformat, granularity = dformat

        output_file_format = "data_{}_{}_{{}}.json".format(dformat, granularity)

        # Get data
        self.to_dformat(dformat)
        vrd_data_train = [self.get_vrd_data_pair(img_path) for img_path in self.splits["train"]]
        vrd_data_test  = [self.get_vrd_data_pair(img_path) for img_path in self.splits["test"]]

        # Granulate if needed
        if granularity == "rel":
          assert dformat == "relst", "Mh. Does it make sense to granulate 'rel' with dformat {}?".format(dformat)
          def granulate(d):
            new_vrd_data = []
            for (img_path, relst) in d:
              if relst is None:
                new_vrd_data.append((None, None))
                continue
              for img_rel in relst:
                new_vrd_data.append((img_path,img_rel))
            return new_vrd_data
          vrd_data_train = granulate(vrd_data_train)
          vrd_data_test  = granulate(vrd_data_test)
        elif granularity == "img":
          pass
        else:
          raise ValueError("Error. Unknown granularity: {}".format(granularity))

        # Save
        self.writejson(vrd_data_train, output_file_format.format("train"))
        self.writejson(vrd_data_test,  output_file_format.format("test"))

    # Transform vrd_data to the desired format
    def to_dformat(self, to_dformat):
      if to_dformat == self.cur_dformat:
        return
      elif self.cur_dformat == "relst" and to_dformat == "annos":
        self.relst2annos()
      elif self.cur_dformat == "annos" and to_dformat == "relst":
        self.annos2relst()
      else:
        raise NotImplementedError("Unknown format conversion: {} -> {}".format(self.cur_dformat, to_dformat))

    # "annos" format separates objects and relationships into two different lists
    def relst2annos(self):
      new_vrd_data = {}
      for img_path, relst in self.vrd_data.items():
        if relst is None:
          new_vrd_data[img_path] = None
          continue
        objects  = []
        rels     = []
        def add_object(obj):
          for i_obj,any_obj in enumerate(objects):
            if any_obj["cls"] == obj["id"] and \
              np.all(any_obj["bbox"] == utils.bboxDictToList(obj["bbox"])):
              return i_obj
          i_obj = len(objects)
          objects.append({"cls" : obj["id"], "bbox" : utils.bboxDictToList(obj["bbox"])})
          return i_obj

        for i_rel, rel in enumerate(relst):
          new_rel = {}
          new_rel["sub"] = add_object(rel["subject"])
          new_rel["obj"] = add_object(rel["object"])
          new_rel["pred"] = rel["predicate"]["id"]
          rels.append(new_rel)

        new_vrd_data[img_path] = {
          "objs" : objects,
          "rels" : rels
        }

      self.vrd_data = new_vrd_data
      self.cur_dformat = "annos"

    def save_counts(self):
      print("\tSaving object and predicate counts...")
      if self.cur_dformat != "relst":
        print("Warning! save_counts requires relst format (I'll convert it, but maybe you want to save_counts later or sooner than now)")
        if not osp.exists(save_dir):
          os.mkdir(save_dir)
        self.to_dformat("relst")

      def get_rels_counts(img_ids, name):
        pred_counts = np.zeros(len(self.pred_vocab), dtype=np.int)
        obj_counts  = np.zeros((len(self.obj_vocab), 3), dtype=np.int) # Count occurrences as sub,obj, and total
        for img_path in img_ids:
          _, relst = self.get_vrd_data_pair(img_path)
          if relst is not None:
            for i_rel, rel in enumerate(relst):
              for id in rel["predicate"]["id"]:
                pred_counts[id]                  += 1
                obj_counts[rel["subject"]["id"],(0,1)] += 1
                obj_counts[rel["object"]["id"],(0,2)]  += 1
        return obj_counts, pred_counts

      obj_counts_train, pred_counts_train  = get_rels_counts(self.splits["train"], "train")
      obj_counts_test,  pred_counts_test   = get_rels_counts(self.splits["test"],  "test")

      if RESORT_PREDICATES_BY_TRAIN_COUNTS:
        print("\tResort predicates by train counts...")
        if self.already_wrote_something:
          print("ERROR! Not a good idea to RESORT_PREDICATES_BY_TRAIN_COUNTS since something has been written already.")
          input()
          exit(0)

        new_inds = pred_counts_train.argsort()[::-1].tolist()

        new_pred_counts_train = pred_counts_train[new_inds]
        new_pred_counts_test = pred_counts_test[new_inds]
        new_pred_vocab = [self.pred_vocab[i] for i in new_inds]
        pred_counts_train = new_pred_counts_train
        pred_counts_test  = new_pred_counts_test
        self.pred_vocab = new_pred_vocab
        for k,relst in self.vrd_data.items():
          if relst is None: continue
          for i_rel,rel in enumerate(relst):
            self.vrd_data[k][i_rel]["predicate"]["id"] = [new_inds.index(x) for x in rel["predicate"]["id"]]
        self.pred_resorted = True

      for set, obj_counts, pred_counts in [("train",obj_counts_train,pred_counts_train),("test",obj_counts_test,pred_counts_test)]:
        self.writejson([(obj,count)  for obj,count  in zip(self.obj_vocab,  obj_counts.tolist())],  "objects-counts_{}.json".format(set))
        self.writejson([(pred,count) for pred,count in zip(self.pred_vocab, pred_counts.tolist())], "predicates-counts_{}.json".format(set))


    def prepareGT(self):
      print("\tGenerating ground-truth pickles and counts...")
      if self.cur_dformat != "relst":
        print("Warning! prepareGT requires relst format (I'll convert it, but maybe you want to prepareGT later or sooner than now)")
        if not osp.exists(save_dir):
          os.mkdir(save_dir)
        self.to_dformat("relst")

      if not osp.exists(self.fullpath("eval", outputdir = True)):
        os.mkdir(self.fullpath("eval", outputdir = True))

      # Output files
      gt_output_path         = osp.join("eval", "gt.pkl")
      gt_zs_output_path      = osp.join("eval", "gt_zs.pkl")

      # Count triplets
      all_train_tuple_labels = []
      all_test_tuple_labels  = []
      all_zs_tuple_labels    = []

      # Count zero-shot objects and predicates
      zs_pred_counts = np.zeros(len(self.pred_vocab), dtype=np.int)
      zs_obj_counts  = np.zeros((len(self.obj_vocab), 3), dtype=np.int) # Count occurrences as sub,obj, and total

      # Test set
      gt_pkl = {}
      gt_pkl["tuple_label"] = []
      gt_pkl["sub_bboxes"]  = []
      gt_pkl["obj_bboxes"]  = []

      # zero-shot set
      gt_zs_pkl = {}
      gt_zs_pkl["tuple_label"] = []
      gt_zs_pkl["sub_bboxes"]  = []
      gt_zs_pkl["obj_bboxes"]  = []

      # Gather tuples in train set (needed for computing zero-shot set)
      for img_path in self.splits["train"]:
        _, relst = self.get_vrd_data_pair(img_path)
        if relst is not None:
          for i_rel, rel in enumerate(relst):
            for id in rel["predicate"]["id"]:
              all_train_tuple_labels.append([rel["subject"]["id"], id, rel["object"]["id"]])

      # Create test/zero-shot set; Count tuples, zero-shot objs/preds
      for img_path in self.splits["test"]:
        _, relst = self.get_vrd_data_pair(img_path)

        tuple_labels = []
        sub_bboxes   = []
        obj_bboxes   = []

        zs_tuple_labels = []
        zs_sub_bboxes   = []
        zs_obj_bboxes   = []

        if relst is not None:
          for i_rel, rel in enumerate(relst):
            # Note: multi-predicate relationship triplets are not allowed in ground-truth pickles; we just list them as multiple triplets with same obj-subj
            for id in rel["predicate"]["id"]:
              tuple_label = [rel["subject"]["id"], id, rel["object"]["id"]]
              sub_bbox = utils.bboxDictToNumpy(rel["subject"]["bbox"])
              obj_bbox = utils.bboxDictToNumpy(rel["object"]["bbox"])

              # test
              all_test_tuple_labels.append(tuple_label)
              tuple_labels.append(tuple_label)
              sub_bboxes.append(sub_bbox)
              obj_bboxes.append(obj_bbox)

              # zero-shot
              if not tuple_label in all_train_tuple_labels:
                zs_tuple_labels.append(tuple_label)
                zs_sub_bboxes.append(sub_bbox)
                zs_obj_bboxes.append(obj_bbox)

                zs_pred_counts[id]                        += 1
                zs_obj_counts[rel["subject"]["id"],(0,1)] += 1
                zs_obj_counts[rel["object"]["id"],(0,2)]  += 1

        # test
        gt_pkl["tuple_label"].append(np.array(tuple_labels, dtype = np.uint8))
        gt_pkl["sub_bboxes"].append(np.array(sub_bboxes,    dtype = np.uint16))
        gt_pkl["obj_bboxes"].append(np.array(obj_bboxes,    dtype = np.uint16))

        # zero-shot
        all_zs_tuple_labels += zs_tuple_labels
        gt_zs_pkl["tuple_label"].append(np.array(zs_tuple_labels, dtype = np.uint8))
        gt_zs_pkl["sub_bboxes"].append(np.array(zs_sub_bboxes,    dtype = np.uint16))
        gt_zs_pkl["obj_bboxes"].append(np.array(zs_obj_bboxes,    dtype = np.uint16))

      # save test/zero-shot gt pickles
      print("\tSaving gt pickles...")
      self.writepickle(gt_pkl, gt_output_path)
      self.writepickle(gt_zs_pkl, gt_zs_output_path)

      # save zero-shot counts
      print("\tSaving zero-shot counts...")
      self.writejson([(obj,count)  for obj,count  in zip(self.obj_vocab,  zs_obj_counts.tolist())],  "objects-counts_{}.json".format("test_zs"))
      self.writejson([(pred,count) for pred,count in zip(self.pred_vocab, zs_pred_counts.tolist())], "predicates-counts_{}.json".format("test_zs"))

      # save tuple counts
      print("\tSaving tuple counts...")
      def get_tuple_counts(tuple_labels_list):
        n_obj = len(self.obj_vocab)
        n_pred = len(self.pred_vocab)
        sop_counts = np.zeros((n_obj, n_obj, n_pred), dtype=np.int)
        for tuple_label in tuple_labels_list:
          subject_label, predicate_labels, object_label = tuple_label
          np.add.at(sop_counts[subject_label][object_label], predicate_labels, 1)
        counts = {}
        for sub_idx in range(n_obj):
          for obj_idx in range(n_obj):
            for pred_idx in range(n_pred):
              if sop_counts[sub_idx][obj_idx][pred_idx] > 0:
                counts[str([sub_idx,obj_idx,pred_idx])] = int(sop_counts[sub_idx][obj_idx][pred_idx])
        return counts
      self.writejson(get_tuple_counts(all_train_tuple_labels), "tuples-counts_{}.json".format("train"))
      self.writejson(get_tuple_counts(all_test_tuple_labels),  "tuples-counts_{}.json".format("test"))
      self.writejson(get_tuple_counts(all_zs_tuple_labels),    "tuples-counts_{}.json".format("test_zs"))

    def annos2relst(self):
      raise NotImplementedError
      self.cur_dformat = "relst"

    def randomize_split(self):
      print("Randomizing splits...")
      indices = self.splits["train"] + self.splits["test"]
      random.shuffle(indices)
      len_train = len(self.splits["train"])
      self.splits = {"train" : indices[:len_train], "test" : indices[len_train:]}

    def readbbox_arr(self, bbox, margin=0):
      return {
        'ymin': (bbox[0]-margin),
        'ymax': (bbox[1]-margin),
        'xmin': (bbox[2]-margin),
        'xmax': (bbox[3]-margin)
      }

    def get_vrd_data_pair(self, k):
      try:
        data = self.vrd_data[k]
        if data is None: return (None, None)
        else: return (k, data)
      except KeyError:
        if k is not None:
          print("Image '{}' not found in train vrd_data (e.g {})".format(k, next(iter(self.vrd_data))))
        return (None, None)

    def needed_idxs(self): return [idx for _,idxs in self.splits.items() for idx in idxs]

    # INPUT/OUTPUT Helpers
    def fullpath(self, filename, outputdir = False):
      if outputdir and hasattr(self, "outdir"): return osp.join(globals.data_dir, self.dir, self.outdir, filename)
      else:                                     return osp.join(globals.data_dir, self.dir, filename)

    # plain files
    def readfile(self, filename):
      return open(self.fullpath(filename), 'r')

    def writefile(self, filename):
      if DRY_RUN: return
      self.already_wrote_something = True
      return open(self.fullpath(filename, outputdir = True), 'w')

    # json files
    def readjson(self, filename):
      with self.readfile(filename) as rfile:
        return json.load(rfile)

    def writejson(self, obj, filename):
      if DRY_RUN: return
      self.already_wrote_something = True
      with self.writefile(filename) as f:
        json.dump(obj, f)

    # pickle files
    def readpickle(self, filename):
      with open(self.fullpath(filename), 'rb') as f:
        return pickle.load(f, encoding="latin1")

    def writepickle(self, obj, filename, to_outputdir = True):
      if DRY_RUN: return
      self.already_wrote_something = True
      with open(self.fullpath(filename, outputdir = to_outputdir), 'wb') as f:
        pickle.dump(obj, f)

    # matlab files
    def readmat(self, filename): return sio.loadmat(self.fullpath(filename))


class VRDPrep(DataPreparer):
    def __init__(self, subset = False, multi_label = True, generate_emb = [], load_dsr = False):
      super(VRDPrep, self).__init__(multi_label = multi_label, generate_emb = generate_emb)

      print("\tVRDPrep(subset : {}, multi-label : {}, generate_emb = {}, load_dsr = {})...".format(subset, multi_label, generate_emb, load_dsr))

      self.dir = "vrd"

      self.train_dsr = self.readpickle("train.pkl")
      self.test_dsr  = self.readpickle("test.pkl")

      self.prepare_vocabs("obj.txt", "rel.txt", subset)

      self.prepare_obj_det_fun = self.prepare_obj_det_FromLP

      # Subset determines subpath directory
      if not load_dsr:
        self.outdir = "all" if subset == False else subset
      else:
        assert subset == False, "Error. Using a subset will cause the vocabs to be different from DSR.".format(subset)
        self.outdir = "dsr"

      # Clean directory
      f = self.fullpath(self.outdir)
      # if os.path.exists(f):
      #   shutil.rmtree(f)
      if not os.path.exists(f):
        os.mkdir(f)

      # Load data
      # TODO: Additionally handle files like {test,train}_image_metadata.json ?
      if load_dsr:
        self.load_dsr()
      else:
        self.load_vrd()

    def load_vrd(self):
      print("\tLoad VRD data...")

      # Read data
      annotations_train = self.readjson("annotations_train.json")
      annotations_test  = self.readjson("annotations_test.json")

      # Join split and transform img filenames to paths
      vrd_data = {}
      for img_file,anns in annotations_train.items(): vrd_data[osp.join("sg_train_images", img_file)] = self._generate_relst(anns)
      for img_file,anns in annotations_test.items():  vrd_data[osp.join("sg_test_images",  img_file)] = self._generate_relst(anns)

      self.vrd_data = vrd_data
      self.cur_dformat = "relst"

      self.splits = {
        "train" : [osp.join(*x["img_path"].split("/")[-2:]) if x is not None else None for x in self.train_dsr],
        "test"  : [osp.join(*x["img_path"].split("/")[-2:]) if x is not None else None for x in self.test_dsr],
      }

      if len(vrd_data) != len(self.needed_idxs()):
        print("\tWarning! vrd_data has a different number of keys than the number of indices in the selected split: {} != {}".format(len(vrd_data), len(self.needed_idxs())))

    def _generate_relst(self, anns):
        self._read_data_check()
        relst = []
        for ann in anns:
            subject_id = self.get_clean_obj_cls(ann['subject']['category'])
            subject_label = self.obj_vocab[subject_id]
            # this is as per the format described in the README of the VRD dataset
            subject_bbox = self.readbbox_arr(ann['subject']['bbox'], 1)

            object_id = self.get_clean_obj_cls(ann['object']['category'])
            object_label = self.obj_vocab[object_id]
            object_bbox = self.readbbox_arr(ann['object']['bbox'], 1)

            predicate_id = self.get_clean_pred_cls(ann['predicate'])
            if predicate_id is None: continue
            #print(self.pred_vocab)
            #print(predicate_id)
            predicate_label = self.pred_vocab[predicate_id]

            rel_data = defaultdict(lambda: dict())
            rel_data['subject']['id']   = subject_id
            rel_data['subject']['name'] = subject_label
            rel_data['subject']['bbox'] = subject_bbox

            rel_data['object']['id']   = object_id
            rel_data['object']['name'] = object_label
            rel_data['object']['bbox'] = object_bbox

            rel_data['predicate']['name'] = [predicate_label]
            rel_data['predicate']['id']   = [predicate_id]

            # Add to the relationships list
            if not self.multi_label:
              if rel_data not in relst:
                relst.append(rel_data)
                # else:
                #     print("Found duplicate relationship in image: {}".format(img_path))
            else:
              found = False
              for i,rel in enumerate(relst):
                if rel_data["subject"] == rel["subject"] and rel_data["object"] == rel["object"] and not rel_data['predicate']['id'][0] in relst[i]['predicate']['id']:
                  if rel_data['predicate']['id'] not in relst[i]['predicate']['id']:
                    relst[i]['predicate']['name'] += rel_data['predicate']['name']
                    relst[i]['predicate']['id']   += rel_data['predicate']['id']
                  found = True
                  break
              if not found:
                  relst.append(rel_data)

        return relst

    ########################################################################################################################
    # Specific functions for converting files from other works (Deep Structural Ranking, Language Priors)
    ########################################################################################################################

    # Create Object Detections from Language Priors
    def prepare_obj_det_FromLP(self):
      print("\tPreparing object detections from Language Priors...")

      if not osp.exists(self.fullpath("eval", outputdir = False)):
        os.mkdir(self.fullpath("eval", outputdir = False))

      det_result_path = osp.join("from-language-priors", "det_result.mat")
      det_result_output_path = osp.join("eval", "det_res.pkl")

      det_result = self.readmat(det_result_path)

      assert len(det_result["detection_bboxes"]) == 1, "ERROR. Malformed .mat file"
      assert len(det_result["detection_labels"]) == 1, "ERROR. Malformed .mat file"
      assert len(det_result["detection_confs"])  == 1, "ERROR. Malformed .mat file"

      det_result_pkl = {}
      det_result_pkl["boxes"] = []
      det_result_pkl["cls"]   = []
      det_result_pkl["confs"] = []

      for i,(lp_boxes, lp_cls, lp_confs) in \
              enumerate(zip(det_result["detection_bboxes"][0],
                            det_result["detection_labels"][0],
                            det_result["detection_confs"][0])):

        # The -1s fixes the matlab-is-1-indexed problem
        transf_lp_boxes = lp_boxes-1
        transf_lp_cls   = self.get_clean_obj_cls(lp_cls-1)
        transf_lp_confs = lp_confs

        det_result_pkl["boxes"].append(np.array(transf_lp_boxes, dtype=np.int))
        det_result_pkl["cls"]  .append(np.array(transf_lp_cls,   dtype=np.int))
        det_result_pkl["confs"].append(np.array(transf_lp_confs, dtype=np.float32))

      # The object detections are shared across all VRD cuts
      self.writepickle(det_result_pkl, det_result_output_path, to_outputdir = False)

    def load_dsr(self):
        """
            This function loads the {train,test}.pkl which contains the original format of the data
            from the vrd-dsr repo, and converts it to the relst format.
        """
        print("\tLoad DSR data...")

        vrd_data = {}
        for (stage, src_data) in [("train", self.train_dsr),("test", self.test_dsr)]:
            for elem in src_data:
                if elem is None:
                    continue
                img_path = osp.join(*elem['img_path'].split("/")[-2:])
                bounding_boxes = elem['boxes']
                subjects = elem['ix1']
                objects = elem['ix2']
                classes = elem['classes']
                relations = elem['rel_classes']
                relationships = []
                for index in range(len(relations)):
                    subject_id = self.get_clean_obj_cls(int(classes[subjects[index]]))
                    subject_label = self.obj_vocab[subject_id]
                    subject_bbox = {
                        'ymin': int(bounding_boxes[subjects[index]][1]),
                        'ymax': int(bounding_boxes[subjects[index]][3]),
                        'xmin': int(bounding_boxes[subjects[index]][0]),
                        'xmax': int(bounding_boxes[subjects[index]][2])
                    }

                    object_id = self.get_clean_obj_cls(int(classes[objects[index]]))
                    object_label = self.obj_vocab[object_id]
                    object_bbox = {
                        'ymin': int(bounding_boxes[objects[index]][1]),
                        'ymax': int(bounding_boxes[objects[index]][3]),
                        'xmin': int(bounding_boxes[objects[index]][0]),
                        'xmax': int(bounding_boxes[objects[index]][2])
                    }

                    # TODO: remove code duplication
                    if not self.multi_label:
                      for predicate_id in relations[index]:
                        predicate_label = self.pred_vocab[predicate_id]

                        rel_data = defaultdict(lambda: dict())
                        rel_data['subject']['id']   = subject_id
                        rel_data['subject']['name'] = subject_label
                        rel_data['subject']['bbox'] = subject_bbox

                        rel_data['object']['id']   = object_id
                        rel_data['object']['name'] = object_label
                        rel_data['object']['bbox'] = object_bbox

                        predicate_id = self.get_clean_pred_cls(int(predicate_id))
                        if predicate_id is None: continue
                        rel_data['predicate']['id']   = [predicate_id]
                        rel_data['predicate']['name'] = predicate_label

                        relationships.append(dict(rel_data))
                    else:
                      predicate_id    = [self.get_clean_pred_cls(int(id)) for id in relations[index]]
                      predicate_id    = [id for id in predicate_id if id is not None]
                      if len(predicate_id) == 0: continue
                      predicate_label = [self.pred_vocab[id] for id in predicate_id]

                      rel_data = defaultdict(lambda: dict())
                      rel_data['subject']['id']   = subject_id
                      rel_data['subject']['name'] = subject_label
                      rel_data['subject']['bbox'] = subject_bbox

                      rel_data['object']['id']   = object_id
                      rel_data['object']['name'] = object_label
                      rel_data['object']['bbox'] = object_bbox

                      rel_data['predicate']['id']   = predicate_id
                      rel_data['predicate']['name'] = predicate_label

                      relationships.append(dict(rel_data))

                vrd_data[img_path] = relationships
                if len(relationships) == 0:
                    print("Note: '{}' has no relationships. ".format(img_path))

        self.vrd_data = vrd_data
        self.cur_dformat = "relst"

        self.splits = {
          "train" : [osp.join(*x["img_path"].split("/")[-2:]) if x is not None else None for x in self.train_dsr],
          "test"  : [osp.join(*x["img_path"].split("/")[-2:]) if x is not None else None for x in self.test_dsr],
        }

    """
    def prepareGT_FromLP(self):
        print("\tPreparing Ground-Truths from Language Priors...")
        '''
            This function creates the pickles used for the evaluation on the for VRD Dataset
            The ground "truths and object detections are provided by Visual Relationships with Language Priors
            (files available on GitHub) as matlab .mat objects.
        '''
        # TODO create object emb json from params_emb.pkl ?

        # Input files
        gt_path         = osp.join("from-language-priors", "gt.mat")
        gt_zs_path      = osp.join("from-language-priors", "zeroShot.mat")

        # Output files
        if not osp.exists(self.fullpath("eval", outputdir = True)):
          os.mkdir(self.fullpath("eval", outputdir = True))
        gt_output_path         = osp.join("eval", "gt.pkl")
        gt_zs_output_path      = osp.join("eval", "gt_zs.pkl")

        gt = self.readmat(gt_path)
        gt_pkl = {}
        gt_pkl["tuple_label"] = gt["gt_tuple_label"][0]-1
        gt_pkl["obj_bboxes"]  = gt["gt_obj_bboxes"][0]
        gt_pkl["sub_bboxes"]  = gt["gt_sub_bboxes"][0]
        self.writepickle(gt_pkl, gt_output_path)

        gt_zs = self.readmat(gt_zs_path)
        zs = gt_zs["zeroShot"][0];
        gt_zs_pkl = deepcopy(gt_pkl)
        for ii in range(len(gt_pkl["tuple_label"])):
          if zs[ii].shape[0] == 0:
            continue
          idx = zs[ii] == 1
          gt_zs_pkl["tuple_label"][ii] = gt_pkl["tuple_label"][ii][idx[0]]
          gt_zs_pkl["obj_bboxes"][ii]  = gt_pkl["obj_bboxes"][ii][idx[0]]
          gt_zs_pkl["sub_bboxes"][ii]  = gt_pkl["sub_bboxes"][ii][idx[0]]
        self.writepickle(gt_zs_pkl, gt_zs_output_path)
    """


class VGPrep(DataPreparer):
    def __init__(self, subset, multi_label=True, generate_emb = []):
        super(VGPrep, self).__init__(multi_label=multi_label, generate_emb=generate_emb)

        print("\tVGPrep(subset : {}, multi-label : {}, generate_emb = {})...".format(subset, multi_label, generate_emb))

        num_objects, num_attributes, num_predicates, subset_map = subset
        self.dir = osp.join("genome", "{}-{}-{}".format(num_objects, num_attributes, num_predicates))

        self.data_format = "json"
        self.img_metadata_file_format = osp.join(self.data_format, "{{}}.{}".format(self.data_format))
        # if the path to metadata files does not exist, generate those files using VGCleaner
        if not osp.exists(self.fullpath(self.data_format)):
          assert DRY_RUN == False, "Can't perform dry run when I need to run VGCleaner()"
          print("\tGenerating {} files for VG relationships...".format(self.data_format))
          cleaner = VGCleaner(num_objects, num_attributes, num_predicates, self.data_format)
          cleaner.build_vocabs_and_json()

        self.prepare_vocabs("objects_vocab.txt", "relations_vocab.txt", subset_map)

        self.outdir = "all" if subset_map == False else subset_map

        # Clean directory
        f = self.fullpath(self.outdir)
        # if os.path.exists(f):
        #   shutil.rmtree(f)
        if not os.path.exists(f):
          os.mkdir(f)

        # LOAD DATA
        print("\tLoad data...")
        self.objects_label_to_id_mapping    = utils.invert_dict(self.obj_vocab)
        self.predicates_label_to_id_mapping = utils.invert_dict(self.pred_vocab)

        self.splits = {
          "train" : [line.split(" ")[0] for line in utils.load_txt_list(self.fullpath("../train.txt"))],
          "test"  : [line.split(" ")[0] for line in utils.load_txt_list(self.fullpath("../val.txt"))],
        }

        vrd_data = {}
        n_not_found = []
        needed_idxs = self.needed_idxs()
        for ix, img_path in enumerate(needed_idxs):
            filepath = self.img_metadata_file_format.format(osp.splitext(osp.basename(img_path))[0])
            try:
              relst = self._generate_relst(self.readjson(filepath))
            except FileNotFoundError:
              relst = None
              n_not_found.append(filepath)

            vrd_data[img_path] = relst

        if len(n_not_found) > 0:
          print("Warning! Couldn't find metadata for {}/{} images. (e.g '{}')".format(len(n_not_found), len(needed_idxs), n_not_found[0]))

        self.vrd_data = vrd_data
        self.cur_dformat = "relst"

    def _generate_relst(self, data):
        self._read_data_check()
        objects_info = {}
        for obj in data['objects']:
            obj_id = self.get_clean_obj_cls(obj['object_id'])
            objects_info[obj_id] = {
                'name': obj['name'][0],
                'bbox': {k: int(v) for k, v in obj['bndbox'].items()}
            }

        relst = []
        for pred in data['relations']:
            subject_info = objects_info[pred['subject_id']]
            object_info  = objects_info[pred['object_id']]
            pred_label   = pred['predicate']

            rel_data = defaultdict(lambda: dict())
            rel_data['subject']['id']   = self.get_clean_obj_cls(int(self.objects_label_to_id_mapping[subject_info['name']]))
            rel_data['subject']['name'] = self.obj_vocab[rel_data['subject']['id']]
            rel_data['subject']['bbox'] = subject_info['bbox']

            rel_data['object']['id']   = self.get_clean_obj_cls(int(self.objects_label_to_id_mapping[object_info['name']]))
            rel_data['object']['name'] = self.obj_vocab[rel_data['object']['id']]
            rel_data['object']['bbox'] = object_info['bbox']
       
            predicate_id = self.get_clean_pred_cls(int(self.predicates_label_to_id_mapping[pred_label]))
            if predicate_id is None: continue
            predicate_label = self.pred_vocab[predicate_id]
            rel_data['predicate']['id'] = [predicate_id]
            rel_data['predicate']['name'] = [predicate_label]

            # Add to the relationships list
            if not self.multi_label:
                if rel_data not in relst:
                    relst.append(rel_data)
            else:
                found = False
                for i, rel in enumerate(relst):
                    if rel_data["subject"] == rel["subject"] and rel_data["object"] == rel["object"] and not rel_data['predicate']['id'][0] in relst[i]['predicate']['id']:
                        if rel_data['predicate']['id'] not in relst[i]['predicate']['id']:
                          relst[i]['predicate']['name'] += rel_data['predicate']['name']
                          relst[i]['predicate']['id']   += rel_data['predicate']['id']
                        found = True
                        break
                if not found:
                  relst.append(rel_data)

        return relst

# Obtain the embedding of a word given the embedding model
def getWordEmbedding(word, emb_model, model_name, depth=0):
  emb_size = globals.emb_model_size(model_name)
  if not hasattr(getWordEmbedding, "fallback_emb_map"):
    # This map defines the fall-back words of words that do not exist in the embedding model
    with open(os.path.join(globals.data_dir, "embeddings", "fallback-v1.json"), 'r') as rfile:
      getWordEmbedding.fallback_emb_map = json.load(rfile)
  try:
    if "glove" in model_name:
      embedding = emb_model.word_vectors[emb_model.dictionary[word]]
    else:
      embedding = emb_model[word]
  except KeyError:
    embedding = np.zeros(emb_size)
    fallback_words = []
    if word in getWordEmbedding.fallback_emb_map:
      fallback_words = getWordEmbedding.fallback_emb_map[word]
    if " " in word:
      fallback_words = ["_".join(word.split(" "))] + fallback_words + [word.split(" ")]

    for fallback_word in fallback_words:
      if isinstance(fallback_word, str):
        embedding = getWordEmbedding(fallback_word, emb_model, model_name, depth=depth+1)
        if np.all(embedding != np.zeros(emb_size)):
          if fallback_word != "_".join(word.split(" ")):
            print("{}'{}' mapped to '{}'".format("  " * depth, word, fallback_word))
          break
      elif isinstance(fallback_word, list):
        fallback_vec = [getWordEmbedding(fb_sw, emb_model, model_name, depth=depth+1) for fb_sw in fallback_word]
        filtered_wv = [(w,v) for w,v in zip(fallback_word,fallback_vec) if not np.all(v == np.zeros(emb_size))]
        fallback_w,fallback_v = [],[]
        if len(filtered_wv) > 0:
          fallback_w,fallback_v = zip(*filtered_wv)
          embedding = np.mean(fallback_v, axis=0)
        if np.all(embedding != np.zeros(emb_size)):
          print("{}'{}' mapped to the average of {}".format("  " * depth, word, fallback_w))
          break
      else:
        raise ValueError("Error fallback word is of type {}: {}".format(fallback_word, type(fallback_word)))
  if np.all(embedding == np.zeros(emb_size)):
    if depth == 0:
      print("{}Warning! Couldn't find semantic vector for '{}'".format("  " * depth, word))
    return embedding
  return embedding / np.linalg.norm(embedding)

# Load embedding model
def load_emb_model(model_name):
  print("Loading embedding model '{}'...".format(model_name))

  # Lookup model in cache
  if not hasattr(load_emb_model, "cache"):
    load_emb_model.cache = {}
  if model_name in load_emb_model.cache:
    return load_emb_model.cache[model_name]

  # Load model
  model_path = globals.emb_model_path(model_name)
  if model_name is "gnews":
    model = KeyedVectors.load_word2vec_format(model_path, binary=True)
  elif "glove" in model_name:
    #raise NotImplementedError
    model = Glove().load(model_path)
    tmp_file = get_tmpfile("{}.txt".format(model_path))
    _ = glove2word2vec(model_path, tmp_file)
    model = Word2Vec.load(tmp_file)
  else:
    # This is needed happens in the case of COCO finetuned models because they were dumped from outside the
    # train_word2vec script, so the train_word2vec module needs to be in the path for them to load
    sys.path.append("./scripts")
    model = VRDEmbedding.load_model(model_path)

  # Cache and return model
  load_emb_model.cache[model_name] = model
  return model


if __name__ == '__main__':

    multi_label = True # False

    generate_embeddings = []
    #generate_embeddings = ["gnews", "50", "100", "coco-70-50", "coco-30-50"]
    #generate_embeddings = ["gnews"]
    #generate_embeddings = ["gnews", "300"]
    #generate_embeddings = ["gnews", "300", "glove-50"]
    #generate_embeddings = ["glove-50"]
    generate_embeddings = ["gnews", "300"] # , "coco-20-300", "coco-50-300", "coco-100-300"]

    """ VRD
    print("Preparing data for VRD")

    vrd_subsets = [False]
    #vrd_subsets = ["spatial", "activities"]
    vrd_subsets = [False, "spatial", "activities"]
    for vrd_subset in vrd_subsets:
      data_preparer_vrd = VRDPrep(subset = vrd_subset, multi_label = multi_label, generate_emb = generate_embeddings)
      data_preparer_vrd.save_data(["relst", "annos"])
    #"""

    #""" VG
    print("Preparing data for VG")
    #subset = (1600, 400, 20) # TODO: allow multi-word vocabs, so that we can load 1600-400-20_bottomup
    vg_subsets = [(150, 50, 50, "all"), (150, 50, 50, "activities"), (150, 50, 50, "spatial")]
    vg_subsets = [(150, 50, 50, "activities"), (150, 50, 50, "spatial")]
    for vg_subset in vg_subsets:
      data_preparer_vg = VGPrep(subset = vg_subset, multi_label=multi_label, generate_emb=generate_embeddings)
      data_preparer_vg.save_data(["relst", "annos"])
      # data_preparer_vg.save_data(["relst", ("relst", "rel"), "annos"])
    #"""


    """ VRD/DSR
    # Generate the data in relst format using the {train,test}.pkl files provided by DSR
    print("Generating data from VRD/DSR")
    data_preparer_vrd = VRDPrep(subset = False, multi_label = multi_label, generate_emb = generate_embeddings, load_dsr = True)
    data_preparer_vrd.save_data(["relst", "annos"])
    #"""

    sys.exit(0)
