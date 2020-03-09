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


# Get subset of vrd_data
def filter_by_subdir(vrd_data, subdir):
    out = defaultdict(lambda: dict())
    for img_path, rel_data in vrd_data.items():
        if img_path.startswith(subdir):
            out[img_path] = rel_data
    return out


def generate_img_relationships(anns, object_mapping, predicate_mapping):
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
        else:
            print("Found duplicate relationship in image: {}".format(img_path))
    
    return relationships


def generate_annotations(anns):
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


if __name__ == '__main__':
    generate_img_rels = True
    objects_vocab_file = "./data/vrd/objects.json"
    predicates_vocab_file = "./data/vrd/predicates.json"
    annotations_train_file = "./data/vrd/annotations_train.json"
    annotations_test_file = "./data/vrd/annotations_test.json"
    if generate_img_rels is True:
        output_file_format = "./data/vrd/data_img_rels_{}.json"
    else:
        output_file_format = "./data/vrd/data_annotations_{}.json"

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
        if generate_img_rels is True:
            vrd_data[img_path] = generate_img_relationships(anns, objects_id_to_label_mapping, predicates_id_to_label_mapping)
        else:
            vrd_data[img_path] = generate_annotations(anns)

    # with open(output_file_format.format(""), 'w') as wfile:
    #     json.dump(vrd_data, wfile)

    with open(output_file_format.format("train"), 'w') as wfile:
        vrd_data_train = filter_by_subdir(vrd_data, "sg_train_images")
        json.dump(vrd_data_train, wfile)

    with open(output_file_format.format("test"), 'w') as wfile:
        vrd_data_test = filter_by_subdir(vrd_data, "sg_test_images")
        json.dump(vrd_data_test, wfile)
