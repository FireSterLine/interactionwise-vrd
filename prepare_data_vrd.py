import os.path as osp
import json
from collections import defaultdict


def generate_mapping(filename):
    id_to_label_mapping = {}
    with open(filename, 'r') as rfile:
        elems = json.load(rfile)

    for index, elem in enumerate(elems):
        id_to_label_mapping[index] = elem

    return id_to_label_mapping

if __name__ == '__main__':

    objects_vocab_file = "./data/vrd/objects.json"
    predicates_vocab_file = "./data/vrd/predicates.json"
    annotations_train_file = "./data/vrd/annotations_train.json"
    annotations_test_file = "./data/vrd/annotations_test.json"
    output_file_format = "./data/vrd/data{}.json"

    objects_id_to_label_mapping = generate_mapping(objects_vocab_file)
    predicates_id_to_label_mapping = generate_mapping(predicates_vocab_file)

    with open(annotations_train_file, 'r') as rfile:
        annotations_train = json.load(rfile)

    with open(annotations_test_file, 'r') as rfile:
        annotations_test = json.load(rfile)

    # Transform img file names to img subpaths
    annotations = {}
    for img_file,anns in annotations_train.items():
        annotations[osp.join("sg_train_images", img_file)] = anns
    for img_file,anns in annotations_test.items():
        annotations[osp.join("sg_test_images", img_file)] = anns

    vrd_data = defaultdict(lambda: list())
    for img_path, anns in annotations.items():
        for ann in anns:
            subject_label = objects_id_to_label_mapping[ann['subject']['category']]
            # this is as per the format described in the README of the VRD dataset
            subject_bbox = {
                'ymin': ann['subject']['bbox'][0],
                'ymax': ann['subject']['bbox'][1],
                'xmin': ann['subject']['bbox'][2],
                'xmax': ann['subject']['bbox'][3]
            }

            object_label = objects_id_to_label_mapping[ann['object']['category']]
            object_bbox = {
                'ymin': ann['object']['bbox'][0],
                'ymax': ann['object']['bbox'][1],
                'xmin': ann['object']['bbox'][2],
                'xmax': ann['object']['bbox'][3]
            }

            predicate_label = predicates_id_to_label_mapping[ann['predicate']]

            rel_data = defaultdict(lambda: dict())
            rel_data['subject']['name'] = subject_label
            rel_data['subject']['id'] = ann['subject']['category']
            rel_data['subject']['bbox'] = subject_bbox

            rel_data['object']['name'] = object_label
            rel_data['object']['id'] = ann['object']['category']
            rel_data['object']['bbox'] = object_bbox

            rel_data['predicate']['name'] = predicate_label
            rel_data['predicate']['id'] = ann['predicate']

            if rel_data not in vrd_data[img_path]:
                vrd_data[img_path].append(rel_data)
            else:
                print("Found duplicate relationship in image: {}".format(img_path))

    with open(output_file_format.format(""), 'w') as wfile:
        json.dump(vrd_data, wfile)

    # Get subset of vrd_data
    def filter_by_subdir(vrd_data, subdir):
        out = defaultdict(lambda: dict())
        for img_path, rel_data in vrd_data.items():
            if img_path.startswith(subdir):
                out[img_path] = rel_data
        return out

    with open(output_file_format.format("train"), 'w') as wfile:
        vrd_data_train = filter_by_subdir(vrd_data, "sg_train_images")
        json.dump(vrd_data_train, wfile)

    with open(output_file_format.format("test"), 'w') as wfile:
        vrd_data_test = filter_by_subdir(vrd_data, "sg_test_images")
        json.dump(vrd_data_test, wfile)
