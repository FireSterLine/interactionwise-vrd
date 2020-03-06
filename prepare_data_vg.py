import json
from glob import glob
from collections import defaultdict


def generate_mapping(filename):
    label_to_id_mapping = {}
    with open(filename, 'r') as rfile:
        for index, line in enumerate(rfile):
            label_to_id_mapping[line.strip()] = index

    return label_to_id_mapping


def generate_data_old_format(data, object_mapping, predicate_mapping):
    objects_info = {}
    for obj in data['objects']:
        obj_vg_id = obj['object_id']
        objects_info[obj_vg_id] = {
            'name': obj['name'][0],
            'bbox': {k: int(v) for k, v in obj['bndbox'].items()}
        }

    relationships = set()
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

        relationships.add(rel_data)
    
    return list(relationships)


def generate_data_new_format(data, object_mapping, predicate_mapping):
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
        if (subject_id, object_id) not in rels.keys():
            rels[str((subject_id, object_id))] = []
        rels[str((subject_id, object_id))].append(pred_id)

    relationships = {
        'objects': objects,
        'relationships': rels
    }
    return relationships


if __name__ == '__main__':
    use_old_format = False
    
    num_objects = 1600
    num_attributes = 400
    num_predicates = 20
    # the training data file created by this script is actually in data/genome/... instead of faster-rcnn/data/genome
    # this is because the faster-rcnn/data/genome has been linked to data/genome
    json_files_path = "./data/genome/{}-{}-{}/json/".format(num_objects, num_attributes, num_predicates)
    objects_vocab_file = "./data/genome/{}-{}-{}/objects_vocab_{}.txt".format(num_objects, num_attributes, num_predicates, num_objects)
    predicates_vocab_file = "./data/genome/{}-{}-{}/relations_vocab_{}.txt".format(num_objects, num_attributes, num_predicates, num_predicates)
    if use_old_format is True:
        output_file = './data/genome/{}-{}-{}/vg_data.json'.format(num_objects, num_attributes, num_predicates)
    else:
        output_file = './data/genome/{}-{}-{}/vg_data_new.json'.format(num_objects, num_attributes, num_predicates)
    
    objects_label_to_id_mapping = generate_mapping(objects_vocab_file)
    predicates_label_to_id_mapping = generate_mapping(predicates_vocab_file)

    relationship_data = defaultdict(lambda: list())
    for ix, filename in enumerate(glob(json_files_path + "*.json")):
        # if ix > 2:
        #     break
        data = json.load(open(filename, 'r'))

        folder = data['folder']
        filename = data['filename']
        img_id = folder + "/" + filename

        if use_old_format is True:
            relationship_data[img_id] = generate_data_old_format(data, objects_label_to_id_mapping, predicates_label_to_id_mapping)
        else:
            relationship_data[img_id] = generate_data_new_format(data, objects_label_to_id_mapping, predicates_label_to_id_mapping)

    json.dump(relationship_data, open(output_file, 'w'))
