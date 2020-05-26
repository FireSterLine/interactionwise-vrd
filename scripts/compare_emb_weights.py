import json
import torch
import pickle
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity


if __name__ == '__main__':
    checkpoint_path = "../data/vrd/epoch_4_checkpoint.pth.tar"
    emb_weights_path = "../data/vrd/params_emb.pkl"
    w2v_model_path = "/home/azfar/PythonProjects/GoogleNews-vectors-negative300.bin.gz"
    vrd_objects_path = "../data/vrd/objects.json"

    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    emb_checkpoint = checkpoint['state_dict']['emb.weight']
    emb_params = pickle.load(open(emb_weights_path, 'rb'), encoding='latin1')

    '''
    print("Loading Word2Vec model...")
    w2v_model = KeyedVectors.load_word2vec_format(w2v_model_path, binary=True)

    print("Loading objects list...")
    objects = json.load(open(vrd_objects_path, 'r'))

    for index, obj in enumerate(objects):
        if ' ' in obj:
            obj = '_'.join(obj.split())
        try:
            w2v_emb = w2v_model[obj].reshape(1, -1)
        except KeyError:
            print("{} not found in w2v model!".format(obj))
            continue
        emb_p = emb_params[index].reshape(1, -1)
        print("\tScale: {}".format(w2v_model[obj]/emb_params[index]))
        # if w2v_emb == emb_p:
        #     print("Match found!")
        print("Cosine similarity for {}: {}".format(obj, cosine_similarity(w2v_emb, emb_p)))
    '''

    similarities = []
    for index, emb_p_vec in enumerate(emb_params):
        emb_p_vec = emb_p_vec.reshape(1, -1)
        for emb_chkpnt in emb_checkpoint:
            emb_chkpnt = emb_chkpnt.reshape(1, -1)
            sim = cosine_similarity(emb_p_vec, emb_chkpnt)
            similarities.append(sim)
            print("Cosine similarity for {}: {}".format(index, cosine_similarity(emb_p_vec, emb_chkpnt)))
    # print("Max similarity score: {}".format(max(similarities)))
    # print("Min similarity score: {}".format(min(similarities)))