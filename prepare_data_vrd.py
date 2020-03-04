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
    annotations_file = "./data/vrd/annotations_train.json"
    output_file = "./data/vrd/vrd_data.json"

    objects_id_to_label_mapping = generate_mapping(objects_vocab_file)
    predicates_id_to_label_mapping = generate_mapping(predicates_vocab_file)

    with open(annotations_file, 'r') as rfile:
        annotations = json.load(rfile)

    relationship_data = defaultdict(lambda: list())
    for img_path, anns in annotations.items():
        for ann in anns:
            subject_label = objects_id_to_label_mapping[ann['subject']['category']]
            subject_bbox = {
                'xmin': ann['subject']['bbox'][0],
                'xmax': ann['subject']['bbox'][1],
                'ymin': ann['subject']['bbox'][2],
                'ymax': ann['subject']['bbox'][3]
            }

            object_label = objects_id_to_label_mapping[ann['object']['category']]
            object_bbox = {
                'xmin': ann['object']['bbox'][0],
                'xmax': ann['object']['bbox'][1],
                'ymin': ann['object']['bbox'][2],
                'ymax': ann['object']['bbox'][3]
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

            relationship_data[img_path].append(rel_data)

    with open(output_file, 'w') as wfile:
        json.dump(relationship_data, wfile)