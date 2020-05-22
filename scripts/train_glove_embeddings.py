import os
import re
import json
import pickle
import joblib
import argparse
import numpy as np
from mittens import Mittens
from glove import Glove, Corpus
from gensim.test.utils import datapath
from gensim.corpora import WikiCorpus

from train_word2vec import VRDEmbedding, EpochLogger, EpochSaver, vrd_tokenize


def train_glove_model(path_prefix, dim, num_epochs, num_cores, server_flag=False):
    vrd_embedder = VRDEmbedding(path_prefix, dim)

    print("Loading wiki dataset...")
    if server_flag:
        wiki_path = os.path.join(path_prefix, "wiki.pkl")
        wiki = pickle.load(open(wiki_path, 'rb'))
    else:
        # FOR TESTING ONLY
        wiki_path = datapath("enwiki-latest-pages-articles1.xml-p000000010p000030302-shortened.bz2")
        wiki = WikiCorpus(wiki_path, tokenizer_func=vrd_embedder.vrd_tokenize)

    # wrap the wiki object in an iterator - this is necessary, otherwise we are unable to index word vectors by word
    #   in the glove model
    print("Wrapping an iterator around the wiki object...")
    wiki_iterator = vrd_embedder.MakeIter(wiki)

    print("Initializing Glove Corpus...")
    corpus = Corpus()
    # window size here is the same as used for training Word2Vec
    print("Creating corpus...")
    corpus.fit(wiki_iterator, window=5)

    print("Initializing Glove model...")
    model = Glove(no_components=dim)
    print("Fitting Glove embeddings on corpus matrix...")
    model.fit(corpus.matrix, epochs=num_epochs, no_threads=num_cores, verbose=True)
    print("Adding Glove dictionary...")
    model.add_dictionary(corpus.dictionary)
    try:
        print("Saving Glove model to disk...")
        model.save(os.path.join(path_prefix, "glove_epoch_{}_dim_{}.model".format(num_epochs, dim)))
    except Exception as e:
        print("The following exception occurred: {}".format(str(e)))
        pass

    print(model.word_vectors[model.dictionary['person']])
    return model


def load_terms_from_text_file(filename):
    terms = []
    with open(filename, 'r') as rfile:
        for line in rfile:
            terms.append(line.lower().strip())
    return terms


def load_terms_fallback_json(filename):
    terms = []
    for key, values in json.load(open(filename, 'r')).items():
        terms.append(key)
        for val in values:
            if type(val) is str:
                terms.append(val)
            elif type(val) is list:
                terms.extend(val)
            else:
                print("This block of code should never be executed")
    return terms


def create_master_list_terms(terms, model):
    def term_exists(term, model):
        try:
            model.word_vectors[model.dictionary[term]]
            return True
        except KeyError:
            return False

    all_terms = []
    for t in terms:
        if len(t.split()) > 1:
            # add underscore-joined term to list of terms if the model knows it
            single_token = '_'.join(t.split())
            if term_exists(single_token, model):
                all_terms.append(single_token)
            # add each individual word that the model knows to list of terms
            for it in t.split():
                if term_exists(it, model):
                    all_terms.append(it)
        else:
            # add the term to the list of terms
            if term_exists(t, model):
                all_terms.append(t)
    return list(set(all_terms))


def fine_tune_embeddings(path_to_model, tokenized_captions_fname, num_epochs, dim):
    print("Loading Glove model...")
    try:
        model = Glove().load(path_to_model)
    except:
        model = joblib.load(path_to_model)

    print("Loading vocabularies, creating master list of terms...")
    all_terms = []
    all_terms.extend(json.load(open("../data/vrd/objects.json", 'r')))
    all_terms.extend(json.load(open("../data/vrd/predicates.json", 'r')))
    all_terms.extend(load_terms_from_text_file("../data/genome/2500-1000-500.old/objects_vocab.txt"))
    all_terms.extend(load_terms_from_text_file("../data/genome/2500-1000-500.old/relations_vocab.txt"))
    all_terms.extend(load_terms_from_text_file("../data/genome/150-50-50/objects_vocab.txt"))
    all_terms.extend(load_terms_from_text_file("../data/genome/150-50-50/relations_vocab.txt"))
    all_terms.extend(load_terms_fallback_json("../data/embeddings/fallback-v1.json"))

    # this is a list of all words we care about from VRD, VG and fallback-json
    all_words = create_master_list_terms(all_terms, model)

    print("Loading COCO captions...")
    tokenized_captions = pickle.load(open(tokenized_captions_fname, 'rb'))

    # get all words from COCO vocabulary
    print("Removing OOV words from COCO captions...")
    coco_words = []
    vocab_tokenized_captions = []
    # remove those words from COCO captions that the model doesn't know of
    for tok_caption in tokenized_captions:
        vocab_caption = []
        for word in tok_caption:
            word = word.lower()
            try:
                model.word_vectors[model.dictionary[word]]
                vocab_caption.append(word)
                # add this word to all_words if it already doesn't exist there
                if word not in all_words:
                    all_words.append(word)
            except KeyError:
                pass
        coco_words.extend(vocab_caption)
        vocab_tokenized_captions.append(vocab_caption)

    # get unique words
    coco_words = list(set(coco_words))

    # add only those words in the vocab and embedding dict which the model knows
    print("Creating COCO vocab...")
    # NOTE: A key idea of finetuning embeddings on the COCO dataset is that out of all terms we care about (which are
    #   are in all_words), only those will be updated which exist in COCO too. Additionlly, all terms that are in COCO
    #   will be updated too, but if they're not in all_terms, we don't particularly care about them
    coco_vocab = []
    model_embeddings = {}
    for word in coco_words:
        try:
            embedding = model.word_vectors[model.dictionary[word]]
            coco_vocab.append(word)
            model_embeddings[word] = embedding
        except KeyError:
            pass

    print("Initializing GloVe corpus with COCO vocab...")
    coco_corpus = Corpus()
    coco_corpus.fit(vocab_tokenized_captions, window=5)

    print("COCO corpus: {}".format(coco_corpus.matrix.shape))
    print("Vocab: {}; first word: {}".format(len(coco_vocab), coco_vocab[0]))
    print("Original embeddings: {}; first word: {}".format(len(model_embeddings), list(model_embeddings.keys())[0]))

    print("Fitting mittens model...")
    mittens_model = Mittens(n=dim, max_iter=num_epochs, display_progress=1)
    new_embeddings = mittens_model.fit(np.array(coco_corpus.matrix.todense()), vocab=coco_vocab,
                                       initial_embedding_dict=model_embeddings)

    person_index = coco_vocab.index('person')
    print("\n'Person' old embedding: {}".format(model.word_vectors[model.dictionary['person']]))
    print("'Person' new embedding: {}".format(new_embeddings[person_index]))

    coco_embeddings = {}
    finetuned_emb_count = 0
    for word in all_words:
        # get the new fine-tuned embedding
        if word in coco_vocab:
            word_index = coco_vocab.index(word)
            coco_embeddings[word] = list(new_embeddings[word_index])
            finetuned_emb_count += 1
        # this word is not originally in COCO and therefore its embedding has not been fine-tuned; get original
        # embedding from the GloVe model trained on Wiki
        else:
            try:
                coco_embeddings[word] = model.word_vectors[model.dictionary[word]]
            except KeyError:
                print("No embedding found for {}!".format(word))

    print("Total terms finetuned on COCO: {}".format(finetuned_emb_count))

    # get the number of epochs the current model is trained on
    epochs_trained = int(re.search(r'(?<=epoch_)\d+', path_to_model).group(0))
    filename = os.path.join(os.path.dirname(path_to_model), 'coco', 'glove_epoch_{}_dim_{}.json'.format(
        epochs_trained + num_epochs,
        dim
    ))
    print("Dumping finetuned embeddings to disk...")
    json.dump(coco_embeddings, open(filename, 'w'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a Word2Vec/FastText model on the Wikipedia dataset")
    parser.add_argument("-SL", "--server_local", dest="server_local", type=str, default="local",
                        help="Define whether the execution environment is the server or local")
    parser.add_argument("-C", "--num_cores", dest="num_cores", type=int, default=6,
                        help="Define the number of cores to use for the training process")
    parser.add_argument("-E", "--num_epochs", dest="num_epochs", type=int, default=5,
                        help="Define the number of epochs to use for the training process")
    parser.add_argument("-D", "--dim", dest="dim", type=int, help="Define the size of the vectors")
    parser.add_argument("-F", "--model_file_name", dest="model_filename", type=str, default=None,
                        help="Define the name of the model to finetune over COCO")
    args = parser.parse_args()

    multi_word_phrases = ["traffic light", "trash can", "next to", "sleep next to", "sit next to", "stand next to",
                          "park next", "walk next to", "stand behind", "sit behind", "park behind", "in the front of",
                          "stand under", "sit under", "walk to", "walk past", "walk beside", "on the top of",
                          "on the left of", "on the right of", "sit on", "stand on", "attach to", "adjacent to",
                          "drive on", "taller than", "park on", "lying on", "lean on", "play with", "sleep on",
                          "outside of", "rest on", "skate on", "banana bunch", "mountain range", "door frame",
                          "tail fin", "telephone pole", "moustache", "train platform", "purple flower", "left ear",
                          "tennis net", "windshield wiper", "bus stop", "lamp shade", "light switch", "shower curtain",
                          "cardboard box", "table cloth", "doughnut", "laptop computer", "parking lot", "guard rail",
                          "tv stand", "traffic signal", "tennis racket", "flower pot", "number 2", "baseball uniform",
                          "fence post", "left hand", "palm tree", "ceiling fan", "clock hand", "lamp post",
                          "light pole", "oven door", "traffic sign", "baseball cap", "tree top", "light bulb",
                          "computer monitor", "door knob", "baseball field", "grass patch", "passenger car",
                          "tennis ball", "window sill", "shower head", "name tag", "front window", "computer mouse",
                          "cutting board", "hind leg", "paper towel", "computer screen", "tissue box", "american flag",
                          "evergreen tree", "tree trunk", "mouse pad", "baseball glove", "minute hand", "window pane",
                          "coffee maker", "front wheel", "road sign", "steering wheel", "tennis player",
                          "manhole cover", "stop light", "street sign", "train station", "brake light", "wine glass"]

    server_flag = False
    if args.server_local.lower().strip() == 'server':
        server_flag = True

    if server_flag:
        path_prefix = "/home/findwise/interactionwise/wikipedia_dump/"
    else:
        path_prefix = "/media/azfar/New Volume/WikiDump/"

    if args.model_filename is None:
        # to train new model
        print("Training new GloVe model over Wiki dataset!")
        model = train_glove_model(path_prefix, args.dim, args.num_epochs, args.num_cores, server_flag)
    else:
        # finetune model
        print("Finetuning existing GloVe model over COCO captions!")
        path_to_model = os.path.join(path_prefix, args.model_filename)
        path_to_captions = "../../coco_captions_tokenized.pkl"
        fine_tune_embeddings(path_to_model, path_to_captions, args.num_epochs, args.dim)
