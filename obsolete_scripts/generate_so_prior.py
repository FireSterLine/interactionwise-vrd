import json
import pickle
import numpy as np
from collections import defaultdict


def generate_mapping(filename):
    label_to_id_mapping = {}
    if filename.split('.')[-1] == 'json':
        with open(filename, 'r') as rfile:
            data = json.load(rfile)
    elif filename.split('.')[-1] == 'txt':
        data = []
        with open(filename, 'r') as rfile:
            for line in rfile:
                data.append(line.strip())

    for index, elem in enumerate(data):
        label_to_id_mapping[elem] = index

    return label_to_id_mapping


if __name__ == '__main__':
    # THIS IS FOR VRD
    # print("Generating the prior for Visual Relationship Dataset...")
    # filename = "data/vrd/vrd_data.json"
    # objects_vocab_filename = "data/vrd/objects.json"
    # predicates_vocab_filename = "data/vrd/predicates.json"
    # output_file = "data/vrd/soP.pkl"

    # THIS IS FOR VG
    print("Generating the prior for Visual Genome Dataset...")
    # filename = "data/genome/vrd_data_1600-400-20.json"
    filename = "data/genome/1600-400-20/vg_data.json"
    objects_vocab_filename = "data/genome/1600-400-20/objects_vocab_1600.txt"
    predicates_vocab_filename = "data/genome/1600-400-20/relations_vocab_20.txt"
    output_file = "data/genome/1600-400-20/soP.pkl"
    
    objects_vocab = generate_mapping(objects_vocab_filename)
    predicates_vocab = generate_mapping(predicates_vocab_filename)
    
    sop_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: int())))

    with open(filename, 'r') as rfile:
        data = json.load(rfile)

    for _, elems in data.items():
        for elem in elems:
            subject_label = elem['subject']['name']
            object_label = elem['object']['name']
            predicate_label = elem['predicate']['name']

            sop_counts[subject_label][object_label][predicate_label] += 1

    print("Number of objects in training data: {}".format(len(sop_counts.keys())))
    print("Number of objects in vocab: {}".format(len(objects_vocab)))
    assert len(sop_counts.keys()) <= len(objects_vocab)

    so_prior = np.zeros((len(objects_vocab), len(objects_vocab), len(predicates_vocab)))

    for out_ix, out_elem in enumerate(objects_vocab.keys()):
        for in_ix, in_elem in enumerate(objects_vocab.keys()):
            total_count = sum(sop_counts[out_elem][in_elem].values())
            if total_count == 0:
                # print("{}-{} doesn't exist!".format(out_elem, in_elem))
                continue
            for p_ix, p_elem in enumerate(predicates_vocab.keys()):
                so_prior[out_ix][in_ix][p_ix] = float(sop_counts[out_elem][in_elem][p_elem]) / float(total_count)

    print(so_prior)
    pickle.dump(so_prior, open(output_file, 'wb'))

    # Computing difference between their so_prior and our so_prior
    # s = pickle.load(open('/home/azfar/Downloads/so_prior.pkl', 'rb'), encoding='latin1')
    
    # total_diff = 0
    # for a in range(100):
    #     for b in range(100):
    #         diff = so_prior[a][b] - s[a][b]
    #         total_diff += max(diff)

    # print(total_diff)
    # TODO: We are getting some difference here, can't figure out why yet.
