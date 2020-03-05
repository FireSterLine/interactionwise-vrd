import json
import pickle
import numpy as np
from collections import defaultdict


def generate_mapping(filename):
    label_to_id_mapping = {}
    with open(filename, 'r') as rfile:
        data = json.load(rfile)

    for index, elem in enumerate(data):
        label_to_id_mapping[elem] = index

    return label_to_id_mapping


if __name__ == '__main__':
    filename = "data/vrd/vrd_data.json"
    objects_vocab_filename = "data/vrd/objects.json"
    predicates_vocab_filename = "data/vrd/predicates.json"
    
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

    assert len(sop_counts.keys()) == len(objects_vocab)

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
    pickle.dump(so_prior, open('data/so_prior_vrd.pkl', 'wb'))

    # Computing difference between their so_prior and our so_prior
    # s = pickle.load(open('/home/azfar/Downloads/so_prior.pkl', 'rb'), encoding='latin1')
    
    # total_diff = 0
    # for a in range(100):
    #     for b in range(100):
    #         diff = so_prior[a][b] - s[a][b]
    #         total_diff += max(diff)

    # print(total_diff)
    # TODO: We are getting some difference here, can't figure out why yet.