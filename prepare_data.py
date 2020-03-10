import sys
import json
from glob import glob
import os.path as osp
from collections import defaultdict


class DataPreparer:
    def prepare_data(self, generate_img_rels):
        pass

    def _generate_mapping(self, filename):
        pass

    def _generate_img_relationships(self, anns, object_mapping, predicate_mapping):
        pass

    def _generate_annotations(self, anns, object_mapping, predicate_mapping):
        pass


class VRDPrep(DataPreparer):
    def __init__(self):
        self.objects_vocab_file = "./data/vrd/objects.json"
        self.predicates_vocab_file = "./data/vrd/predicates.json"
        self.annotations_train_file = "./data/vrd/annotations_train.json"
        self.annotations_test_file = "./data/vrd/annotations_test.json"

        with open("data/vrd/train.pkl", 'rb') as f:
            self.train_dsr = pickle.load(f, encoding="latin1")

        with open("data/vrd/test.pkl", 'rb') as f:
            self.test_dsr = pickle.load(f, encoding="latin1")


    def prepare_data(self, generate_img_rels):
        if generate_img_rels is True:
            output_file_format = "./data/vrd/data_img_rels_{}.json"
        else:
            output_file_format = "./data/vrd/data_annotations_{}.json"

        objects_id_to_label_mapping = self._generate_mapping(self.objects_vocab_file)
        predicates_id_to_label_mapping = self._generate_mapping(self.predicates_vocab_file)

        with open(self.annotations_train_file, 'r') as rfile:
            annotations_train = json.load(rfile)

        with open(self.annotations_test_file, 'r') as rfile:
            annotations_test = json.load(rfile)

        # Transform img file names to img subpaths
        annotations = {}
        for img_file,anns in annotations_train.items():
            annotations[osp.join("sg_train_images", img_file)] = anns
        for img_file,anns in annotations_test.items():
            annotations[osp.join("sg_test_images", img_file)] = anns

        vrd_data = defaultdict(lambda: list())
        for img_path, anns in annotations.items():
            if generate_img_rels is True:
                vrd_data[img_path] = self._generate_img_relationships(anns, objects_id_to_label_mapping, predicates_id_to_label_mapping)
            else:
                vrd_data[img_path] = self._generate_annotations(anns)

        with open(output_file_format.format("train"), 'w') as wfile:
            vrd_data_train = self._filter_by_subdir(vrd_data, "sg_train_images")

            # Reorder such that images are ordered same as dsr
            vrd_data_train_sorted = []
            for i in self.train_dsr:
                for im_path in vrd_data_train:
                  if i["filename"] in k:
                    break
                vrd_data_train_sorted.append((im_path, vrd_data[im_path]))

            json.dump(vrd_data_train_sorted, wfile)

        with open(output_file_format.format("test"), 'w') as wfile:
            vrd_data_test = self._filter_by_subdir(vrd_data, "sg_test_images")

            # Reorder such that images are ordered same as dsr
            vrd_data_test_sorted = []
            for i in self.test_dsr:
                for im_path in vrd_data_test:
                  if i["filename"] in k:
                    break
                vrd_data_test_sorted.append((im_path, vrd_data[im_path]))

            json.dump(vrd_data_test_sorted, wfile)

    def _generate_mapping(self, filename):
        id_to_label_mapping = {}
        with open(filename, 'r') as rfile:
            elems = json.load(rfile)

        for index, elem in enumerate(elems):
            id_to_label_mapping[index] = elem

        return id_to_label_mapping

    # Get subset of vrd_data
    def _filter_by_subdir(self, vrd_data, subdir):
        out = defaultdict(lambda: dict())
        for img_path, rel_data in vrd_data.items():
            if img_path.startswith(subdir):
                out[img_path] = rel_data
        return out

    def _generate_img_relationships(self, anns, object_mapping, predicate_mapping):
        relationships = []
        for ann in anns:
            subject_id = ann['subject']['category']
            subject_label = object_mapping[subject_id]
            # this is as per the format described in the README of the VRD dataset
            subject_bbox = {
                'ymin': ann['subject']['bbox'][0],
                'ymax': ann['subject']['bbox'][1],
                'xmin': ann['subject']['bbox'][2],
                'xmax': ann['subject']['bbox'][3]
            }

            object_id = ann['object']['category']
            object_label = object_mapping[object_id]
            object_bbox = {
                'ymin': ann['object']['bbox'][0],
                'ymax': ann['object']['bbox'][1],
                'xmin': ann['object']['bbox'][2],
                'xmax': ann['object']['bbox'][3]
            }

            predicate_id = ann['predicate']
            predicate_label = predicate_mapping[predicate_id]

            rel_data = defaultdict(lambda: dict())
            rel_data['subject']['name'] = subject_label
            rel_data['subject']['id'] = subject_id
            rel_data['subject']['bbox'] = subject_bbox

            rel_data['object']['name'] = object_label
            rel_data['object']['id'] = object_id
            rel_data['object']['bbox'] = object_bbox

            rel_data['predicate']['name'] = predicate_label
            rel_data['predicate']['id'] = predicate_id

            if rel_data not in relationships:
                relationships.append(rel_data)
            # else:
            #     print("Found duplicate relationship in image: {}".format(img_path))

        return relationships

    def _generate_annotations(self, anns):
        objects = {}
        rels = {}
        for ann in anns:
            subject_id = int(ann['subject']['category'])
            # this is as per the format described in the README of the VRD dataset
            subject_bbox = {
                'ymin': ann['subject']['bbox'][0],
                'ymax': ann['subject']['bbox'][1],
                'xmin': ann['subject']['bbox'][2],
                'xmax': ann['subject']['bbox'][3]
            }

            object_id = int(ann['object']['category'])
            object_bbox = {
                'ymin': ann['object']['bbox'][0],
                'ymax': ann['object']['bbox'][1],
                'xmin': ann['object']['bbox'][2],
                'xmax': ann['object']['bbox'][3]
            }

            predicate_id = int(ann['predicate'])

            objects[subject_id] = subject_bbox
            objects[object_id] = object_bbox

            rel_key = str((subject_id, object_id))
            if rel_key not in rels.keys():
                rels[rel_key] = set()
            rels[rel_key].add(predicate_id)

        # convert the sets to lists before returning
        rels = {k: list(v) for k, v in rels.items()}

        relationships = {
            'objects': objects,
            'relationships': rels
        }
        return relationships


class VGPrep(DataPreparer):
    def __init__(self):
        self.num_objects = 1600
        self.num_attributes = 400
        self.num_predicates = 20
        # the training data file created by this script is actually in data/genome/... instead of faster-rcnn/data/genome
        # this is because the faster-rcnn/data/genome has been linked to data/genome
        self.json_files_path = "./data/genome/{}-{}-{}/json/".format(self.num_objects, self.num_attributes, self.num_predicates)
        self.objects_vocab_file = "./data/genome/{}-{}-{}/objects_vocab_{}.txt".format(self.num_objects, self.num_attributes, self.num_predicates, self.num_objects)
        self.predicates_vocab_file = "./data/genome/{}-{}-{}/relations_vocab_{}.txt".format(self.num_objects, self.num_attributes, self.num_predicates, self.num_predicates)

    def prepare_data(self, generate_img_rels):
        # this format is used for generating the so_prior
        if generate_img_rels is True:
            output_file = './data/genome/{}-{}-{}/data_img_rels.json'.format(self.num_objects, self.num_attributes, self.num_predicates)
        # this format is used for training the model
        else:
            output_file = './data/genome/{}-{}-{}/data_annotations.json'.format(self.num_objects, self.num_attributes, self.num_predicates)

        objects_label_to_id_mapping = self._generate_mapping(self.objects_vocab_file)
        predicates_label_to_id_mapping = self._generate_mapping(self.predicates_vocab_file)

        relationship_data = defaultdict(lambda: list())
        for ix, filename in enumerate(glob(self.json_files_path + "*.json")):
            # if ix > 2:
            #     break
            data = json.load(open(filename, 'r'))

            folder = data['folder']
            filename = data['filename']
            img_id = folder + "/" + filename

            if generate_img_rels is True:
                relationship_data[img_id] = self._generate_img_relationships(data, objects_label_to_id_mapping, predicates_label_to_id_mapping)
            else:
                relationship_data[img_id] = self._generate_annotations(data, objects_label_to_id_mapping, predicates_label_to_id_mapping)

        json.dump(relationship_data, open(output_file, 'w'))

    def _generate_mapping(self, filename):
        label_to_id_mapping = {}
        with open(filename, 'r') as rfile:
            for index, line in enumerate(rfile):
                label_to_id_mapping[line.strip()] = index

        return label_to_id_mapping


    def _generate_img_relationships(self, data, object_mapping, predicate_mapping):
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


    def _generate_annotations(self, data, object_mapping, predicate_mapping):
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
    # select the dataset to generate the data for. This can either be 'vrd' or 'vg'
    dataset = 'vrd'
    # select whether to generate image_rels or annotations
    # if true, image_rels will be generated
    # if false, annotations will be generated
    generate_img_rels = True
    if dataset.lower().strip() == 'vrd':
        obj = VRDPrep()
    else:
        obj = VGPrep()

    obj.prepare_data(generate_img_rels)
