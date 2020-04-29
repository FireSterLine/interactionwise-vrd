import os
import sys
import os.path as osp
from glob import glob

from collections import defaultdict

import numpy as np
import scipy.io as sio

from gensim.models import KeyedVectors
from gensim.models import Word2Vec

import json
import pickle
import globals
import utils
import random
from copy import deepcopy

from data.genome.clean_vg import VGCleaner
from scripts.train_word2vec import VRDEmbedding, EpochLogger, EpochSaver

# Prepare the data without saving anything at all
DRY_RUN = False

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
    def __init__(self, multi_label = True, generate_emb = None):
        self.objects_vocab_file    = "objects.json"
        self.predicates_vocab_file = "predicates.json"

        self.obj_vocab    = None
        self.pred_vocab   = None

        self.multi_label  = multi_label

        self.generate_emb = generate_emb

        self.dir         = None
        self.vrd_data    = None
        self.cur_dformat = None
        self.splits      = None
        self.prefix      = "data"

        self.to_delete = ["soP.pkl", "pP.pkl"]

    # This function reads the dataset's vocab txt files and loads them
    def prepare_vocabs(self, obj_vocab_file, pred_vocab_file, use_cleaning_map = False):
        obj_vocab = utils.load_txt_list(self.fullpath(obj_vocab_file))
        pred_vocab = utils.load_txt_list(self.fullpath(pred_vocab_file))

        if use_cleaning_map is not False:
          cleaning_map = self.readjson("cleaning_map.json")
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
          # assert cleaning_map["obj_map"] == {}, "NotImplemented: A cleaning map for objects requires some preprocessing to the object_detections as well, if not a re-train of the object detection model. Better not touch these things."
          # obj_vocab,  obj_cls_map  = cleaned_vocab(obj_vocab,  cleaning_map["obj_map"])
          pred_vocab, pred_cls_map = cleaned_vocab(pred_vocab, cleaning_map["pred_map"], cleaning_map["pred_subset"])
          self.cleaning_map = {"pred" : pred_cls_map}

        self.writejson(obj_vocab,  self.objects_vocab_file)
        self.writejson(pred_vocab, self.predicates_vocab_file)
        self.obj_vocab  = obj_vocab
        self.pred_vocab = pred_vocab

        if self.generate_emb is not None:
          for model_name,model in self.generate_emb.items():
            obj_emb  = [ getWordEmbedding(obj_label,  model, globals.emb_model_size(model_name)).astype(float).tolist() for  obj_label in self.obj_vocab]
            pred_emb = [ getWordEmbedding(pred_label, model, globals.emb_model_size(model_name)).astype(float).tolist() for pred_label in self.pred_vocab]
            self.writejson(obj_emb,  "objects-emb-{}.json".format(model_name))
            self.writejson(pred_emb, "predicates-emb-{}.json".format(model_name))


    # This function converts to relst
    def _generate_relst(self, anns): raise NotImplementedError
    def _generate_annos(self, anns): raise NotImplementedError

    # Save data
    def save_data(self, dformat, granularity = "img"):
        print("\tGenerating '{}' data with '{}' granularity...".format(dformat, granularity))
        self.to_dformat(dformat)

        for to_delete in self.to_delete:
          if os.path.exists(self.fullpath(to_delete)):
            os.remove(self.fullpath(to_delete))

        output_file_format = "{}_{}_{}_{{}}.json".format(self.prefix, dformat, granularity)

        vrd_data_train = []
        vrd_data_test = []
        for img_path in self.splits["train"]:
          vrd_data_train.append(self.get_vrd_data_pair(img_path))
        for img_path in self.splits["test"]:
          vrd_data_test.append(self.get_vrd_data_pair(img_path))

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

        self.writejson(vrd_data_train, output_file_format.format("train"))
        self.writejson(vrd_data_test,  output_file_format.format("test"))

    # transform vrd_data to the desired format
    def to_dformat(self, to_dformat):
      if to_dformat == self.cur_dformat:
        return
      elif self.cur_dformat == "relst" and to_dformat == "annos":
        self.relst2annos()
      elif self.cur_dformat == "annos" and to_dformat == "relst":
        self.annos2relst()
      else:
        raise NotImplementedError("Unknown format conversion: {} -> {}".format(self.cur_dformat, to_dformat))

    # "annos" format separates objects and relatinoships
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

    def prepareGT(self):
      print("\tGenerating ground-truth pickle...")
      if self.cur_dformat != "relst":
        print("Warning! prepareGT requires relst format (I'll convert it, but maybe you want to prepareGT later or sooner than now)")
        if not osp.exists(save_dir):
          os.mkdir(save_dir)
        self.to_dformat("relst")

      # Output files
      if not osp.exists(self.fullpath("eval")):
        os.mkdir(self.fullpath("eval"))
      gt_output_path         = osp.join("eval", "gt.pkl")
      # TODO: zeroshot
      gt_zs_output_path      = osp.join("eval", "gt_zs.pkl")

      if os.path.exists(self.fullpath(gt_output_path)):
        os.remove(self.fullpath(gt_output_path))
      if os.path.exists(self.fullpath(gt_zs_output_path)):
        os.remove(self.fullpath(gt_zs_output_path))

      gt_pkl = {}
      gt_pkl["tuple_label"] = []
      gt_pkl["sub_bboxes"]  = []
      gt_pkl["obj_bboxes"]  = []

      # TODO: zeroshot: count all the different triples in the train set, and filter out the ones in the test set that are also in the train set
      gt_zs_pkl = {}
      gt_zs_pkl["tuple_label"] = []
      gt_zs_pkl["sub_bboxes"]  = []
      gt_zs_pkl["obj_bboxes"]  = []
      train_tuple_label = []

      for img_path in self.splits["train"]:
        _, relst = self.get_vrd_data_pair(img_path)
        if relst is not None:
          for i_rel, rel in enumerate(relst):
            for id in rel["predicate"]["id"]:
              train_tuple_label.append([rel["subject"]["id"], id, rel["object"]["id"]])

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
            # multi_label (namely multi-predicate relationships) are not allowed in ground-truth pickles
            for id in rel["predicate"]["id"]:
              tuple_label = [rel["subject"]["id"], id, rel["object"]["id"]]
              sub_bbox = utils.bboxDictToNumpy(rel["subject"]["bbox"])
              obj_bbox = utils.bboxDictToNumpy(rel["object"]["bbox"])

              tuple_labels.append(tuple_label)
              sub_bboxes.append(sub_bbox)
              obj_bboxes.append(obj_bbox)
              if not tuple_label in train_tuple_label:
                zs_tuple_labels.append(tuple_label)
                zs_sub_bboxes.append(sub_bbox)
                zs_obj_bboxes.append(obj_bbox)

        tuple_labels = np.array(tuple_labels, dtype = np.uint8)
        sub_bboxes   = np.array(sub_bboxes,   dtype = np.uint16)
        obj_bboxes   = np.array(obj_bboxes,   dtype = np.uint16)

        zs_tuple_labels = np.array(zs_tuple_labels, dtype = np.uint8)
        zs_sub_bboxes   = np.array(zs_sub_bboxes,   dtype = np.uint16)
        zs_obj_bboxes   = np.array(zs_obj_bboxes,   dtype = np.uint16)

        gt_pkl["tuple_label"].append(tuple_labels)
        gt_pkl["sub_bboxes"].append(sub_bboxes)
        gt_pkl["obj_bboxes"].append(obj_bboxes)

        gt_zs_pkl["tuple_label"].append(zs_tuple_labels)
        gt_zs_pkl["sub_bboxes"].append(zs_sub_bboxes)
        gt_zs_pkl["obj_bboxes"].append(zs_obj_bboxes)

      #gt_pkl    = np.array(gt_pkl)
      #gt_zs_pkl = np.array(gt_zs_pkl)

      self.writepickle(gt_pkl, gt_output_path)
      self.writepickle(gt_zs_pkl, gt_zs_output_path)


    # def annos2relst(self):
    #     TODO
    #     self.cur_dformat = "relst"

    def get_vrd_data_pair(self, k):
      try:
        data = self.vrd_data[k]
        if data is None: return (None, None)
        else: return (k, data)
      except KeyError:
        if k is not None:
          print("Image '{}' not found in train vrd_data (e.g {})".format(k, next(iter(self.vrd_data))))
        return (None, None)

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

    # INPUT/OUTPUT Helpers
    def fullpath(self, filename):
      return osp.join(globals.data_dir, self.dir, filename)

    # plain files
    def readfile(self, filename):
      return open(self.fullpath(filename), 'r')

    def writefile(self, filename):
      if DRY_RUN: return
      return open(self.fullpath(filename), 'w')

    # json files
    def readjson(self, filename):
      with self.readfile(filename) as rfile:
        return json.load(rfile)

    def writejson(self, obj, filename):
      if DRY_RUN: return
      with self.writefile(filename) as f:
        json.dump(obj, f)

    # pickle files
    def readpickle(self, filename):
      with open(self.fullpath(filename), 'rb') as f:
        return pickle.load(f, encoding="latin1")

    def writepickle(self, obj, filename):
      if DRY_RUN: return
      with open(self.fullpath(filename), 'wb') as f:
        pickle.dump(obj, f)

    # matlab files
    def readmat(self, filename): return sio.loadmat(self.fullpath(filename))


class VRDPrep(DataPreparer):
    def __init__(self, multi_label = True, generate_emb = None, use_cleaning_map = False):
      super(VRDPrep, self).__init__(multi_label = multi_label, generate_emb = generate_emb)

      print("\tVRDPrep(multi-label : {}, use_cleaning_map : {})...".format(multi_label, use_cleaning_map))

      self.dir = "vrd"
      self.use_cleaning_map = use_cleaning_map

      self.train_dsr = self.readpickle("train.pkl")
      self.test_dsr  = self.readpickle("test.pkl")

      self.prepare_vocabs("obj.txt", "rel.txt", use_cleaning_map = self.use_cleaning_map)

      # TODO: Additionally handle files like {test,train}_image_metadata.json

    def get_clean_obj_cls(self,  cls):
      return cls # if not (hasattr(self, "cleaning_map") and "obj" in self.cleaning_map) else self.cleaning_map["obj"][cls]
    def get_clean_pred_cls(self, cls):
      if (hasattr(self, "cleaning_map") and "pred" in self.cleaning_map):
        # print(self.cleaning_map["pred"])
        return self.cleaning_map["pred"].get(cls, None)
      else:
        return cls

    def load_vrd(self):
      print("\tLoad VRD data...")
      # LOAD DATA
      annotations_train = self.readjson("annotations_train.json")
      annotations_test  = self.readjson("annotations_test.json")

      # Transform img file names to img subpaths
      annotations = {}
      for img_file,anns in annotations_train.items():
        annotations[osp.join("sg_train_images", img_file)] = anns
      for img_file,anns in annotations_test.items():
        annotations[osp.join("sg_test_images", img_file)] = anns

      vrd_data = {}
      for img_path, anns in annotations.items():
          vrd_data[img_path] = self._generate_relst(anns)

      self.vrd_data = vrd_data
      self.cur_dformat = "relst"

      self.splits = {
        "train" : [osp.join(*x["img_path"].split("/")[-2:]) if x is not None else None for x in self.train_dsr],
        "test"  : [osp.join(*x["img_path"].split("/")[-2:]) if x is not None else None for x in self.test_dsr],
      }

    def _generate_relst(self, anns):
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
                  relst[i]['predicate']['name'] += rel_data['predicate']['name']
                  relst[i]['predicate']['id']   += rel_data['predicate']['id']
                  found = True
                  break
              if not found:
                  relst.append(rel_data)

        return relst

    def load_dsr(self):
        """
            This function loads the {train,test}.pkl which contains the original format of the data
            from the vrd-dsr repo, and converts it to the relst format.
        """
        print("\tLoad dsr VRD data...")
        assert self.use_cleaning_map == False, "Error. use_cleaning_map will cause the vocabs to be different from DSR."

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

        self.prefix = "dsr"

    def prepareEvalFromLP(self):
        '''
            This function creates the pickles used for the evaluation on the for VRD Dataset
            The ground "truths and object detections are provided by Visual Relationships with Language Priors
            (files available on GitHub) as matlab .mat objects.
        '''

        # Input files
        det_result_path = osp.join("eval", "from-language-priors", "det_result.mat")
        gt_path         = osp.join("eval", "from-language-priors", "gt.mat")
        gt_zs_path      = osp.join("eval", "from-language-priors", "zeroShot.mat")

        # Output files
        det_result_output_path = osp.join("eval", "det_res.pkl")
        gt_output_path         = osp.join("eval", "gt.pkl")
        gt_zs_output_path      = osp.join("eval", "gt_zs.pkl")

        def prepareGT():
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

        def prepareDetRes():
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
              transf_lp_cls   = lp_cls-1
              transf_lp_confs = lp_confs

              det_result_pkl["boxes"].append(np.array(transf_lp_boxes, dtype=np.int))
              det_result_pkl["cls"]  .append(np.array(transf_lp_cls,   dtype=np.int))
              det_result_pkl["confs"].append(np.array(transf_lp_confs, dtype=np.float32))

          self.writepickle(det_result_pkl, det_result_output_path)

        prepareDetRes()
        prepareGT()


class VGPrep(DataPreparer):
    def __init__(self, subset, multi_label=True, generate_emb=None):
        super(VGPrep, self).__init__(multi_label=multi_label, generate_emb=generate_emb)

        print("\tLoad VRD data...")
        num_objects, num_attributes, num_predicates = subset
        self.dir = osp.join("genome", "{}-{}-{}".format(num_objects, num_attributes, num_predicates))

        self.data_format = "json"
        self.img_metadata_file_format = osp.join(self.data_format, "{{}}.{}".format(self.data_format))
        # if the path to metadata files does not exist, generate those files using VGCleaner
        if not osp.exists(self.fullpath(self.data_format)):
            assert DRY_RUN == False, "Can't perform dry run when I need to run VGCleaner()"
            print("Generating {} files for VG relationships...".format(self.data_format))
            cleaner = VGCleaner(num_objects, num_attributes, num_predicates, self.data_format)
            cleaner.build_vocabs_and_json()

        self.prepare_vocabs("objects_vocab.txt", "relations_vocab.txt")

        # LOAD DATA
        self.objects_label_to_id_mapping = utils.invert_dict(self.obj_vocab)
        self.predicates_label_to_id_mapping = utils.invert_dict(self.pred_vocab)
        # print(self.predicates_label_to_id_mapping)
        # print(self.objects_label_to_id_mapping)

        self.splits = {
            "train" : [line.split(" ")[0] for line in utils.load_txt_list(self.fullpath("../train.txt"))],
            "test"  : [line.split(" ")[0] for line in utils.load_txt_list(self.fullpath("../val.txt"))],
        }
        needed_idxs = [idx for _,idxs in self.splits.items() for idx in idxs]

        vrd_data = {}
        n_not_found = []
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
        objects_info = {}
        for obj in data['objects']:
            obj_id = obj['object_id']
            objects_info[obj_id] = {
                'name': obj['name'][0],
                'bbox': {k: int(v) for k, v in obj['bndbox'].items()}
            }

        relst = []
        for pred in data['relations']:
            subject_info = objects_info[pred['subject_id']]
            object_info = objects_info[pred['object_id']]
            pred_label = pred['predicate']

            rel_data = defaultdict(lambda: dict())
            rel_data['subject']['name'] = subject_info['name']
            rel_data['subject']['id'] = self.objects_label_to_id_mapping[subject_info['name']]
            rel_data['subject']['bbox'] = subject_info['bbox']

            rel_data['object']['name'] = object_info['name']
            rel_data['object']['id'] = self.objects_label_to_id_mapping[object_info['name']]
            rel_data['object']['bbox'] = object_info['bbox']

            rel_data['predicate']['name'] = [pred_label]
            rel_data['predicate']['id'] = [self.predicates_label_to_id_mapping[pred_label]]

            # Add to the relationships list
            if not self.multi_label:
                if rel_data not in relst:
                    relst.append(rel_data)
            else:
                found = False
                for i, rel in enumerate(relst):
                    if rel_data["subject"] == rel["subject"] and rel_data["object"] == rel["object"] and not rel_data['predicate']['id'][0] in relst[i]['predicate']['id']:
                        relst[i]['predicate']['name'] += rel_data['predicate']['name']
                        relst[i]['predicate']['id']   += rel_data['predicate']['id']
                        found = True
                        break
                if not found:
                  relst.append(rel_data)

        return relst

def getWordEmbedding(word, emb_model, emb_size, depth=0):
    if not hasattr(getWordEmbedding, "fallback_emb_map"):
        # This map defines the fall-back words of words that do not exist in the embedding model
        with open(os.path.join(globals.data_dir, "embeddings", "fallback-v1.json"), 'r') as rfile:
            getWordEmbedding.fallback_emb_map = json.load(rfile)
    try:
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
            embedding = getWordEmbedding(fallback_word, emb_model, depth+1)
            if np.all(embedding != np.zeros(emb_size)):
              if fallback_word != "_".join(word.split(" ")):
                print("{}'{}' mapped to '{}'".format("  " * depth, word, fallback_word))
              break
          elif isinstance(fallback_word, list):
            fallback_vec = [getWordEmbedding(fb_sw, emb_model, depth+1) for fb_sw in fallback_word]
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
      print("{}Warning! Couldn't find semantic vector for '{}'".format("  " * depth, word))
      return embedding
    return embedding / np.linalg.norm(embedding)




if __name__ == '__main__':

    # TODO: filter out relationships between the same object?

    multi_label = True # False

    # generate_embeddings = False
    generate_embeddings = ["gnews", "50", "100", "coco-30-50"]

    w2v_model = None
    if generate_embeddings:
      w2v_model = {}
      for model_name in generate_embeddings:
        print("Loading embedding model '{}'...".format(model_name))
        model_path = globals.emb_model_path(model_name)
        if model_name is "gnews":
          model = KeyedVectors.load_word2vec_format(model_path, binary=True)
        else:
          # This is needed happens in the case of COCO finetuned models because they were dumped from outside the
          # train_word2vec script, so the train_word2vec module needs to be in the path for them to load
          sys.path.append("./scripts")
          model = VRDEmbedding.load_model(model_path)
        w2v_model[model_name] = model

    #"""
    print("Preparing data for VRD!")
    data_preparer_vrd = VRDPrep(use_cleaning_map=True, multi_label=multi_label, generate_emb=w2v_model)
    #"""
    #print("\tPreparing evaluation data from Language Priors...")
    #data_preparer_vrd.prepareEvalFromLP()
    data_preparer_vrd.load_vrd()
    #data_preparer_vrd.randomize_split()
    data_preparer_vrd.prepareGT()
    data_preparer_vrd.save_data("relst")
    #data_preparer_vrd.save_data("relst", "rel")
    data_preparer_vrd.save_data("annos")
    #"""

    """
    # Generate the data in relst format using the {train,test}.pkl files provided by DSR
    print("Generating data in DSR format...")
    data_preparer_vrd.load_dsr()
    data_preparer_vrd.save_data("relst")
    #data_preparer_vrd.save_data("relst", "rel")
    data_preparer_vrd.save_data("annos")
    """

    """
    # TODO: allow multi-word vocabs, so that we can load 1600-400-20_bottomup
    print("Preparing data for VG...")
    subset = (150, 50, 50)
    #subset = (1600, 400, 20)
    data_preparer_vg = VGPrep(subset, multi_label=multi_label, generate_emb=w2v_model)
    data_preparer_vg.save_data("relst")
    data_preparer_vg.prepareGT()
    #data_preparer_vg.save_data("relst", "rel")
    data_preparer_vg.save_data("annos")
    #"""
