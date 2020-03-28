import sys
import os.path as osp
from glob import glob

from collections import defaultdict

import numpy as np
import scipy.io as sio

import json
import pickle
import globals
import utils

class DataPreparer:
    def __init__(self):
        self.objects_vocab_file    = "objects.json"
        self.predicates_vocab_file = "predicates.json"

    def prepare_data(self, generate_img_rels, granularity):
        pass

    def _generate_img_rels(self, anns, object_mapping, predicate_mapping):
        pass

    def _generate_anno(self, anns, object_mapping, predicate_mapping):
        pass

    def load_txt_vocab(self, filename):
        vocab = []
        with self.readfile(filename) as rfile:
            for index, line in enumerate(rfile):
                vocab.append(line.strip())
        return vocab

    def readbbox(self, bbox, margin=0):
        return {
            'ymin': (bbox[0]-margin),
            'ymax': (bbox[1]-margin),
            'xmin': (bbox[2]-margin),
            'xmax': (bbox[3]-margin)
        }

    # INPUT/OUTPUT Helpers
    def fullpath(self, filename):
        return osp.join(globals.data,_dir, self.metadata_subfolder, filename)

    # plain files
    def readfile(self, filename):
        return open(self.fullpath(filename), 'r')
    def writefile(self, filename):
        return open(self.fullpath(filename), 'w')
    # json files
    def readjson(self, filename):
        with self.readfile(filename) as rfile:
            return json.load(rfile)
        return open(self.fullpath(filename), 'w')
    def writejson(self, filename):
        with self.writefile(filename) as f:
            json.dump(rfile, f)
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
    def __init__(self):
        super(DataPreparer, self).__init__()

        self.metadata_subfolder = "vrd"

        self.train_dsr = self.readpickle("train.pkl")
        self.test_dsr  = self.readpickle("test.pkl")

    def prepare_vocabs(self):
        self.writejson(self.load_txt_vocab("obj.txt"), self.objects_vocab_file)
        self.writejson(self.load_txt_vocab("rel.txt"), self.predicates_vocab_file)

    def convert_train_test_dsr_img_rels(self, type='train'):
        '''
            This function loads the {train,test}.pkl which contains the original format of the data
            from the vrd-dsr repo, and converts it to the img_rels format (same as the one generated)
            by the _generate_img_rels function.
        '''
        objects_id_to_label_mapping    = self.readjson(self.objects_vocab_file)
        predicates_id_to_label_mapping = self.readjson(self.predicates_vocab_file)

        vrd_data = []
        if type == 'train':
            data = self.train_dsr
        elif type == 'test':
            data = self.test_dsr
        else:
            return
        for elem in data:
            if elem is None:
                vrd_data.append((None, None))
                continue
            img_path = "/".join(elem['img_path'].split("/")[-2:])
            bounding_boxes = elem['boxes']
            subjects = elem['ix1']
            objects = elem['ix2']
            classes = elem['classes']
            relations = elem['rel_classes']
            relationships = []
            for index in range(len(relations)):
                subject_id = classes[subjects[index]]
                subject_label = objects_id_to_label_mapping[subject_id]
                subject_bbox = {
                    'ymin': int(bounding_boxes[subjects[index]][1]),
                    'ymax': int(bounding_boxes[subjects[index]][3]),
                    'xmin': int(bounding_boxes[subjects[index]][0]),
                    'xmax': int(bounding_boxes[subjects[index]][2])
                }

                object_id = classes[objects[index]]
                object_label = objects_id_to_label_mapping[object_id]
                object_bbox = {
                    'ymin': int(bounding_boxes[objects[index]][1]),
                    'ymax': int(bounding_boxes[objects[index]][3]),
                    'xmin': int(bounding_boxes[objects[index]][0]),
                    'xmax': int(bounding_boxes[objects[index]][2])
                }

                for pred in relations[index]:
                    predicate_id = pred
                    predicate_label = predicates_id_to_label_mapping[predicate_id]

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

            if len(relationships) > 0:
                vrd_data.append((img_path, relationships))
            else:
                print(img_path)

        print(vrd_data[0])
        self.writejson("dsr_img_rels_{}.json".format(type), vrd_data)



    def prepare_data(self, generate_img_rels, granularity):

        if generate_img_rels is True:
            output_file_format = "data_img_rels_{}_{{}}.json".format(granularity)
        else:
            output_file_format = "data_anno_{}_{{}}.json".format(granularity)

        objects_id_to_label_mapping    = self.readjson(self.objects_vocab_file)
        predicates_id_to_label_mapping = self.readjson(self.predicates_vocab_file)

        annotations_train = self.readjson("annotations_train.json")
        annotations_test  = self.readjson("annotations_test.json")

        # Transform img file names to img subpaths
        annotations = {}
        for img_file,anns in annotations_train.items():
            annotations[osp.join("sg_train_images", img_file)] = anns
        for img_file,anns in annotations_test.items():
            annotations[osp.join("sg_test_images", img_file)] = anns

        vrd_data = defaultdict(lambda: list())
        for img_path, anns in annotations.items():
            if generate_img_rels is True:
                vrd_data[img_path] = self._generate_img_rels(anns, objects_id_to_label_mapping, predicates_id_to_label_mapping)
            else:
                vrd_data[img_path] = self._generate_anno(anns)


        vrd_data_train = self._filter_by_subdir(vrd_data, "sg_train_images")
        vrd_data_test  = self._filter_by_subdir(vrd_data, "sg_test_images")

        def _reorder_as(vrd_data, vrd_data_order_source):
            # Reorder such that images are ordered same as dsr
            vrd_data_sorted = []
            for i in vrd_data_order_source:
                if i is None:
                    vrd_data_sorted.append((None, None))
                    continue
                # this loop is to iterate through the data until we get to the element that is the current loop
                # element - this is to ensure that the order remains the same
                found = False
                for im_path in vrd_data:
                    if osp.basename(i["img_path"]) in im_path:
                        found = True
                        break
                if not found:
                    raise ValueError("Error! Couldn't find {} in order source!".format(i["img_path"]))
                if granularity == 'rel':
                    for elem in vrd_data[im_path]:
                        vrd_data_sorted.append((im_path, elem))
                elif granularity == 'img':
                    vrd_data_sorted.append((im_path, vrd_data[im_path]))
                else:
                    raise ValueError("Error. Unknown granularity: {}".format(granularity))
                del vrd_data[im_path]

            print(len(vrd_data_sorted))
            return vrd_data_sorted

        # Reorder them as DSR pickle
        # TODO: instead, load it as dictionary and load the ids of train and test?
        self.writejson(_reorder_as(vrd_data_train, self.train_dsr), output_file_format.format("train"))
        self.writejson(_reorder_as(vrd_data_test,  self.test_dsr),  output_file_format.format("test"))

    # Get subset of vrd_data
    def _filter_by_subdir(self, vrd_data, subdir):
        out = defaultdict(lambda: dict())
        for img_path, rel_data in vrd_data.items():
            if img_path.startswith(subdir):
                out[img_path] = rel_data
        return out

    # An "img_rels" is something like [rel1, rel2, ...]
    def _generate_img_rels(self, anns, object_mapping, predicate_mapping):
        relationships = []
        for ann in anns:
            subject_id = ann['subject']['category']
            subject_label = object_mapping[subject_id]
            # this is as per the format described in the README of the VRD dataset
            subject_bbox = self.readbbox(ann['subject']['bbox'], 1)

            object_id = ann['object']['category']
            object_label = object_mapping[object_id]
            object_bbox = self.readbbox(ann['object']['bbox'], 1)

            predicate_id = ann['predicate']
            predicate_label = predicate_mapping[predicate_id]

            rel_data = defaultdict(lambda: dict())
            rel_data['subject']['id']   = subject_id
            rel_data['subject']['name'] = subject_label
            rel_data['subject']['bbox'] = subject_bbox

            rel_data['object']['id']   = object_id
            rel_data['object']['name'] = object_label
            rel_data['object']['bbox'] = object_bbox

            rel_data['predicate']['name'] = predicate_label
            rel_data['predicate']['id'] = predicate_id

            if rel_data not in relationships:
                relationships.append(rel_data)
            # else:
            #     print("Found duplicate relationship in image: {}".format(img_path))

        return relationships

    # An "anno" is something like {"objects" : [obj1, obj2, ...], "relationships" : [rel1, rel2, ...]}
    def _generate_anno(self, anns):
        objects = {}
        rels = {}
        for ann in anns:
            subject_id = int(ann['subject']['category'])
            # this is as per the format described in the README of the VRD dataset
            subject_bbox = self.readbbox(ann['subject']['bbox'], 1)

            object_id = int(ann['object']['category'])
            object_bbox = self.readbbox(ann['object']['bbox'], 1)

            predicate_id = int(ann['predicate'])

            objects[subject_id] = subject_bbox
            objects[object_id] = object_bbox

            rel_key = str((subject_id, object_id))
            if rel_key not in rels.keys():
                rels[rel_key] = set()
            rels[rel_key].add(predicate_id)

        # convert the sets to lists before returning
        rels = {k: list(v) for k, v in rels.items()}

        annotations = {
            'objects' : objects,
            'relationships' : rels
        }
        return annotations

    # This function creates the pickles used for the evaluation on the for VRD Dataset
    # The ground truths and object detections are provided by Visual Relationships with Language Priors (files available on GitHub)
    #  as matlab .mat objects.
    def prepareEvalFromLP(self):

        # Input files
        det_result_path = osp.join("eval", "from-language-priors", "det_result.mat")
        gt_path         = osp.join("eval", "from-language-priors", "gt.mat")
        gt_zs_path      = osp.join("eval", "from-language-priors", "zeroShot.mat")

        # Output files
        det_result_output_path = osp.join("eval", "det_res.pkl")
        gt_output_path         = osp.join("eval", "gt.pkl")
        gt_zs_output_path      = osp.join("eval", "gt_zs.pkl")


        det_result = self.readmat(det_result_path)
        # TODO
        #gt         = self.readmat(gt_path)
        #gt_zs      = self.readmat(gt_zs_path)

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

class VGPrep(DataPreparer):
    def __init__(self):
        super(DataPreparer, self).__init__()

        self.num_objects    = 1600
        self.num_attributes = 400
        self.num_predicates = 20

        self.metadata_subfolder = osp.join("genome", "{}-{}-{}".format(self.num_objects, self.num_attributes, self.num_predicates))

        # TODO?: self.train_dsr = self.readpickle("train.pkl")
        # TODO?: self.test_dsr  = self.readpickle("test.pkl")

        self.json_selector = osp.join("json", "*.json")

    def prepare_vocabs(self):
        self.writejson(self.load_txt_vocab("objects_vocab.txt"),   self.objects_vocab_file)
        self.writejson(self.load_txt_vocab("relations_vocab.txt"), self.predicates_vocab_file)

    def prepare_data(self, generate_img_rels, granularity):
        # TODO: granularity
        # this format is used for generating the so_prior
        if generate_img_rels is True:
            output_file = 'data_img_rels.json'
        else:
            output_file = 'data_anno.json'

        objects_label_to_id_mapping = utils.invert_dict(self.readjson(self.objects_vocab_file))
        predicates_label_to_id_mapping = utils.invert_dict(self.readjson(self.predicates_vocab_file))

        relationship_data = defaultdict(lambda: list())
        for ix, filename in enumerate(glob(self.fullpath(self.json_selector))):
            # if ix > 2:
            #     break
            data = self.readjson(filename)

            img_id = osp.join(data['folder'], data['filename'])

            if generate_img_rels is True:
                relationship_data[img_id] = self._generate_img_rels(data, objects_label_to_id_mapping, predicates_label_to_id_mapping)
            else:
                relationship_data[img_id] = self._generate_anno(data, objects_label_to_id_mapping, predicates_label_to_id_mapping)

        self.writejson(relationship_data, output_file)

    def _generate_img_rels(self, data, object_mapping, predicate_mapping):
        objects_info = {}
        for obj in data['objects']:
            obj_vg_id = obj['object_id']
            objects_info[obj_vg_id] = {
                'name': obj['name'][0],
                'bbox': {k: int(v) for k, v in obj['bndbox'].items()}
            }

        relationships = []
        for pred in data['relations']:
            subject_info = objects_info[pred['subject_id']]
            object_info = objects_info[pred['object_id']]
            pred_label = pred['predicate']

            rel_data = defaultdict(lambda: dict())
            rel_data['subject']['name'] = subject_info['name']
            rel_data['subject']['id'] = object_mapping[subject_info['name']]
            rel_data['subject']['bbox'] = subject_info['bbox']

            rel_data['object']['name'] = object_info['name']
            rel_data['object']['id'] = object_mapping[object_info['name']]
            rel_data['object']['bbox'] = object_info['bbox']

            rel_data['predicate']['name'] = pred_label
            rel_data['predicate']['id'] = predicate_mapping[pred_label]

            rel_data = dict(rel_data)
            if rel_data not in relationships:
                relationships.append(rel_data)

        return relationships

    def _generate_anno(self, data, object_mapping, predicate_mapping):
        objects = {}
        obj_id_to_class_id_mapping = {}
        for obj in data['objects']:
            class_id = object_mapping[obj['name'][0]]
            objects[class_id] = {k: int(v) for k, v in obj['bndbox'].items()}
            obj_id_to_class_id_mapping[obj['object_id']] = class_id

        rels = {}
        for pred in data['relations']:
            subject_id = obj_id_to_class_id_mapping[pred['subject_id']]
            object_id = obj_id_to_class_id_mapping[pred['object_id']]
            pred_id = predicate_mapping[pred['predicate']]

            # do we want a set of this?
            rel_key = str((subject_id, object_id))
            if rel_key not in rels.keys():
                rels[rel_key] = set()
            rels[rel_key].add(pred_id)

        # convert the sets to lists before returning
        rels = {k: list(v) for k, v in rels.items()}

        relationships = {
            'objects': objects,
            'relationships': rels
        }
        return relationships


if __name__ == '__main__':

    # select whether to generate image_rels or annotations
    # if true,  image_rels  will be generated
    # if false, annotations will be generated
    generate_img_rels = True

    data_preparer_vrd = VRDPrep()
    data_preparer_vrd.prepare_vocabs()
    data_preparer_vrd.prepareEvalFromLP()

    # Generate the data in img_rels format using the {train,test}.pkl files provided by DSR
    data_preparer_vrd.convert_train_test_dsr_img_rels(type='train')
    data_preparer_vrd.convert_train_test_dsr_img_rels(type='test')


    data_preparer_vg  = VGPrep()
    data_preparer_vg.prepare_vocabs()


    # This is to generate the data in img_rels format using the original annotations in VRD
    # If batching is set to True, each relationship within an image will be a separate instance, as
    # opposed to a set of relationships within an image instance. This is so to facilitate proper
    # batching via the PyTorch Dataloader
    data_preparer.prepare_data(generate_img_rels, granularity='rel')
