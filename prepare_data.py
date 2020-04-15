import os.path as osp
from glob import glob

from collections import defaultdict

import numpy as np
import scipy.io as sio

# from gensim.models import KeyedVectors
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
from gensim.test.utils import get_tmpfile

import json
import pickle
import globals
import utils
from copy import deepcopy

from data.genome.clean_vg import VGCleaner

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
    def __init__(self, multi_label=True, generate_emb=None):
        self.objects_vocab_file = "objects.json"
        self.predicates_vocab_file = "predicates.json"

        self.obj_vocab = None
        self.pred_vocab = None

        self.multi_label = multi_label

        self.generate_emb = generate_emb

        self.dir = None
        self.vrd_data = None
        self.cur_dformat = None
        self.splits = None
        self.prefix = "data"

    # This function reads the dataset's vocab txt files and loads them
    def prepare_vocabs(self, obj_vocab_file, pred_vocab_file):
        self.writejson(utils.load_txt_list(self.fullpath(obj_vocab_file)),  self.objects_vocab_file)
        self.writejson(utils.load_txt_list(self.fullpath(pred_vocab_file)), self.predicates_vocab_file)
        self.obj_vocab  = self.readjson(self.objects_vocab_file)
        self.pred_vocab = self.readjson(self.predicates_vocab_file)

        if self.generate_emb is not None:
            obj_emb = [ utils.getEmbedding(obj_label,  self.generate_emb).astype(float).tolist() for  obj_label in self.obj_vocab]
            pred_emb = [ utils.getEmbedding(pred_label, self.generate_emb).astype(float).tolist() for pred_label in self.pred_vocab]
            self.writejson(obj_emb,  "objects-emb.json")
            self.writejson(pred_emb, "predicates-emb.json")


    # This function converts to relst
    def _generate_relst(self, anns): pass
    def _generate_annos(self, anns): pass

    # Save data
    def save_data(self, dformat, granularity = "img"):
        self.to_dformat(dformat)

        output_file_format = "{}_{}_{}_{{}}.json".format(self.prefix, dformat, granularity)

        vrd_data_train = []
        vrd_data_test = []
        for img_path in self.splits['train']:
          vrd_data_train.append(self.get_vrd_data_pair(img_path))
        for img_path in self.splits['test']:
          vrd_data_test.append(self.get_vrd_data_pair(img_path))

        if granularity == "rel":
          assert dformat == "relst", "Mh. Does it make sense to granulate 'rel' with dformat {}?".format(dformat)
          def granulate(d):
            new_vrd_data = []
            for (img_path, relst) in d:
              if img_path is None:
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
      if self.cur_dformat != "relst":
        print("prepareGT requires relst format (I'll convert it, but maybe you want to prepareGT later or sooner than now)")
        if not osp.exists(save_dir):
          os.mkdir(save_dir)
        self.to_dformat("relst")

      # Output files
      if not osp.exists(self.fullpath("eval")):
        os.mkdir(self.fullpath("eval"))
      gt_output_path         = osp.join("eval", "gt.pkl")
      # TODO: zeroshot
      # gt_zs_output_path      = osp.join("eval", "gt_zs.pkl")

      gt_pkl = {}
      gt_pkl["tuple_label"] = []
      gt_pkl["sub_bboxes"]  = []
      gt_pkl["obj_bboxes"]  = []

      for img_path in self.splits['test']:
        _, relst = self.get_vrd_data_pair(img_path)

        if relst is None:
          tuple_label = None
          sub_bboxes  = None
          obj_bboxes  = None
        else:
          tuple_label = []
          sub_bboxes  = []
          obj_bboxes  = []

          for i_rel, rel in enumerate(relst):
            # multi_label (namely multi-predicate relationships) are not allowed in ground-truth pickles
            for id in rel["predicate"]["id"]:
              tuple_label.append([rel["subject"]["id"], id, rel["object"]["id"]])
              sub_bboxes.append(rel["subject"]["bbox"])
              obj_bboxes.append(rel["object"]["bbox"])

        gt_pkl["tuple_label"].append(tuple_label)
        gt_pkl["sub_bboxes"].append(sub_bboxes)
        gt_pkl["obj_bboxes"].append(obj_bboxes)

      self.writepickle(gt_pkl, gt_output_path)

      # TODO: zeroshot
      # self.writepickle(gt_zs_pkl, gt_zs_output_path)


    # def annos2relst(self):
    #     TODO
    #     self.cur_dformat = "relst"

    def get_vrd_data_pair(self, k):
      try:
        return (k, self.vrd_data[k])
      except KeyError:
        if k is not None:
          print("Image '{}' not found in train vrd_data (e.g {})".format(k, next(iter(self.vrd_data))))
        return (None, None)


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
        return open(self.fullpath(filename), 'w')

    # json files
    def readjson(self, filename):
        with self.readfile(filename) as rfile:
            return json.load(rfile)

    def writejson(self, obj, filename):
        with self.writefile(filename) as f:
            json.dump(obj, f)

    # pickle files
    def readpickle(self, filename):
        with open(self.fullpath(filename), 'rb') as f:
            return pickle.load(f, encoding="latin1")

    def writepickle(self, obj, filename):
        with open(self.fullpath(filename), 'wb') as f:
            pickle.dump(obj, f)

    # matlab files
    def readmat(self, filename): return sio.loadmat(self.fullpath(filename))


class VRDPrep(DataPreparer):
    def __init__(self, multi_label = True, generate_emb = None):
        super(VRDPrep, self).__init__(multi_label = multi_label, generate_emb = generate_emb)

        self.dir = "vrd"

        self.train_dsr = self.readpickle("train.pkl")
        self.test_dsr  = self.readpickle("test.pkl")

        self.prepare_vocabs("obj.txt", "rel.txt")

        # TODO: Additionally handle files like {test,train}_image_metadata.json

    def load_vrd(self):
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
            subject_id = ann['subject']['category']
            subject_label = self.obj_vocab[subject_id]
            # this is as per the format described in the README of the VRD dataset
            subject_bbox = self.readbbox_arr(ann['subject']['bbox'], 1)

            object_id = ann['object']['category']
            object_label = self.obj_vocab[object_id]
            object_bbox = self.readbbox_arr(ann['object']['bbox'], 1)

            predicate_id = ann['predicate']
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
                    subject_id = classes[subjects[index]]
                    subject_label = self.obj_vocab[subject_id]
                    subject_bbox = {
                        'ymin': int(bounding_boxes[subjects[index]][1]),
                        'ymax': int(bounding_boxes[subjects[index]][3]),
                        'xmin': int(bounding_boxes[subjects[index]][0]),
                        'xmax': int(bounding_boxes[subjects[index]][2])
                    }

                    object_id = classes[objects[index]]
                    object_label = self.obj_vocab[object_id]
                    object_bbox = {
                        'ymin': int(bounding_boxes[objects[index]][1]),
                        'ymax': int(bounding_boxes[objects[index]][3]),
                        'xmin': int(bounding_boxes[objects[index]][0]),
                        'xmax': int(bounding_boxes[objects[index]][2])
                    }

                    if not self.multi_label:
                      for predicate_id in relations[index]:
                        predicate_label = self.pred_vocab[predicate_id]

                        rel_data = defaultdict(lambda: dict())
                        rel_data['subject']['id'] = int(subject_id)
                        rel_data['subject']['name'] = subject_label
                        rel_data['subject']['bbox'] = subject_bbox

                        rel_data['object']['id'] = int(object_id)
                        rel_data['object']['name'] = object_label
                        rel_data['object']['bbox'] = object_bbox

                        rel_data['predicate']['id'] = int(predicate_id)
                        rel_data['predicate']['name'] = predicate_label

                        relationships.append(dict(rel_data))
                    else:
                      pred_ids = relations[index]
                      predicate_id    = [int(id)             for id in pred_ids]
                      predicate_label = [self.pred_vocab[id] for id in pred_ids]

                      rel_data = defaultdict(lambda: dict())
                      rel_data['subject']['id']   = int(subject_id)
                      rel_data['subject']['name'] = subject_label
                      rel_data['subject']['bbox'] = subject_bbox

                      rel_data['object']['id']   = int(object_id)
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

        num_objects, num_attributes, num_predicates = subset
        self.dir = osp.join("genome", "{}-{}-{}".format(num_objects, num_attributes, num_predicates))

        self.data_format = "json"
        self.json_selector = osp.join(self.data_format, "*.{}".format(self.data_format))
        # if the path to JSON files does not exist, generate those files using VGCleaner
        if osp.exists(self.fullpath(self.data_format)) is False:
            print("Generating {} files for VG relationships...".format(self.data_format))
            cleaner = VGCleaner(num_objects, num_attributes, num_predicates, self.data_format)
            cleaner.build_vocabs_and_json()

        self.prepare_vocabs("objects_vocab.txt", "relations_vocab.txt")

        # LOAD DATA
        self.objects_label_to_id_mapping = utils.invert_dict(self.obj_vocab)
        self.predicates_label_to_id_mapping = utils.invert_dict(self.pred_vocab)
        # print(self.predicates_label_to_id_mapping)
        # print(self.objects_label_to_id_mapping)

        vrd_data = {}
        for ix, filename in enumerate(glob(self.fullpath(self.json_selector))):

            # NOTE: glob outputs the whole path relative to the current directory
            data = self.readjson("/".join(filename.split("/")[-2:]))

            img_path = osp.join(data['folder'], data['filename'])
            # print(filename, img_path)
            vrd_data[img_path] = self._generate_relst(data)

        self.vrd_data = vrd_data
        self.cur_dformat = "relst"

        self.splits = {
            "train" : [line.split(" ")[0] for line in utils.load_txt_list(self.fullpath("../train.txt"))],
            "test"  : [line.split(" ")[0] for line in utils.load_txt_list(self.fullpath("../test.txt"))],
        }

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


class EpochSaver(CallbackAny2Vec):
    def __init__(self, path_prefix):
        self.path_prefix = path_prefix
        self.epoch = 0

    def on_epoch_end(self, model):
        print("Saving checkpoint {}".format(self.epoch))
        output_path = get_tmpfile("{}epoch_{}.model".format(self.path_prefix, self.epoch))
        model.save(output_path)
        # remove previously saved checkpoint for storage purposes
        prev_checkpoint = "{}epoch_{}.model".format(self.path_prefix, self.epoch - 1)
        if os.path.exists(prev_checkpoint):
            print("Removing previous checkpoint...")
            os.remove(prev_checkpoint)
        self.epoch += 1


class EpochLogger(CallbackAny2Vec):
    def __init__(self):
        self.epoch = 0

    def on_epoch_begin(self, model):
        print("Starting epoch # {}".format(self.epoch))

    def on_epoch_end(self, model):
        print("Ending epoch {}".format(self.epoch))
        print("----------------------")
        self.epoch += 1


if __name__ == '__main__':

    # TODO: filter out relationships between the same object?

    multi_label = False
    generate_embeddings = False

    w2v_model = None
    if generate_embeddings:
        # w2v_model = KeyedVectors.load_word2vec_format(osp.join(globals.data_dir, globals.w2v_model_path), binary=True)
        w2v_model = Word2Vec.load(globals.w2v_model_path)

    print("Preparing data for VRD!")
    data_preparer_vrd = VRDPrep(multi_label=multi_label, generate_emb = w2v_model)
    print("\tPreparing evaluation data from Language Priors...")
    data_preparer_vrd.prepareEvalFromLP()
    print("\tLoad VRD data...")
    data_preparer_vrd.load_vrd()
    print("\tGenerating data in relst format...")
    data_preparer_vrd.save_data("relst")
    # print("\tGenerating ground truth data...")
    #data_preparer_vrd.prepareGT()
    print("\tGenerating data in relst format at relationship level...")
    data_preparer_vrd.save_data("relst", "rel")
    print("\tGenerating data in annos format...")
    data_preparer_vrd.save_data("annos")

    """
    # Generate the data in relst format using the {train,test}.pkl files provided by DSR
    print("Generating data in DSR format...")
    data_preparer_vrd.load_dsr()
    data_preparer_vrd.save_data("relst")
    data_preparer_vrd.save_data("relst", "rel")
    data_preparer_vrd.save_data("annos")
    """
    """
    # TODO: test to see if VG preparation is valid
    # TODO: allow multi-word vocabs, so that we can load 1600-400-20_bottomup
    print("Preparing data for VG...")
    data_preparer_vg = VGPrep((150, 50, 50), multi_label=multi_label, generate_emb=w2v_model)
    # data_preparer_vg  = VGPrep((1600, 400, 20), multi_label=multi_label, generate_emb = w2v_model)
    print("\tGenerating relst data with granularity img...")
    data_preparer_vg.save_data("relst")
    print("\tGenerating ground-truth pickle...")
    data_preparer_vg.prepareGT()
    print("\tGenerating relst data with granularity rel...")
    data_preparer_vg.save_data("relst", "rel")
    print("\tGenerating annos data...")
    data_preparer_vg.save_data("annos")
    """
