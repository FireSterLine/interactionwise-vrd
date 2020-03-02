import json
from glob import glob
from collections import defaultdict


def generate_mapping(filename):
    label_to_id_mapping = {}
    with open(filename, 'r') as rfile:
        for index, line in enumerate(rfile):
            label_to_id_mapping[line.strip()] = index

    return label_to_id_mapping


if __name__ == '__main__':
    num_objects = 1600
    num_attributes = 400
    num_predicates = 20
    # the training data file created by this script is actually in data/genome/... instead of faster-rcnn/data/genome
    # this is because the faster-rcnn/data/genome has been linked to data/genome
    json_files_path = "./faster-rcnn/data/genome/{}-{}-{}/json/".format(num_objects, num_attributes, num_predicates)
    objects_vocab_file = "./faster-rcnn/data/genome/{}-{}-{}/objects_vocab_{}.txt".format(num_objects, num_attributes, num_predicates, num_objects)
    predicates_vocab_file = "./faster-rcnn/data/genome/{}-{}-{}/relations_vocab_{}.txt".format(num_objects, num_attributes, num_predicates, num_predicates)
    output_file = './faster-rcnn/data/genome/vrd_data_{}-{}-{}.json'.format(num_objects, num_attributes, num_predicates)
    save_bounding_boxes = False

    objects_label_to_id_mapping = generate_mapping(objects_vocab_file)
    predicates_label_to_id_mapping = generate_mapping(predicates_vocab_file)

    relationship_data = defaultdict(lambda: defaultdict(lambda: list()))
    for filename in glob(json_files_path + "*.json"):
        data = json.load(open(filename, 'r'))

        objects_vg_id_to_label = {}
        for obj in data['objects']:
            obj_vg_id = obj['object_id']
            objects_vg_id_to_label[obj_vg_id] = {'name': obj['name'][0]}
            # if save_bounding_boxes is True:

        folder = data['folder']
        filename = data['filename']
        for pred in data['relations']:
            subject_label = objects_vg_id_to_label[pred['subject_id']]
            object_label = objects_vg_id_to_label[pred['object_id']]
            pred_label = pred['predicate']

            rel_data = defaultdict(lambda: dict())
            rel_data['subject']['name'] = subject_label['name']
            rel_data['subject']['id'] = objects_label_to_id_mapping[subject_label['name']]

            rel_data['object']['name'] = object_label['name']
            rel_data['object']['id'] = objects_label_to_id_mapping[object_label['name']]

            rel_data['predicate']['name'] = pred_label
            rel_data['predicate']['id'] = predicates_label_to_id_mapping[pred_label]

            if rel_data not in relationship_data[folder][filename]:
                relationship_data[folder][filename].append(rel_data)

    json.dump(relationship_data, open(output_file, 'w'))
