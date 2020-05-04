import os
import pickle
import argparse
from glove import Glove, Corpus
from gensim.test.utils import datapath
from gensim.corpora import WikiCorpus

from train_word2vec import VRDEmbedding, EpochLogger, EpochSaver, vrd_tokenize


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a Word2Vec/FastText model on the Wikipedia dataset")
    parser.add_argument("-SL", "--server_local", dest="server_local", type=str, default="local",
                        help="Define whether the execution environment is the server or local")
    parser.add_argument("-C", "--num_cores", dest="num_cores", type=int, default=6,
                        help="Define the number of cores to use for the training process")
    parser.add_argument("-E", "--num_epochs", dest="num_epochs", type=int, default=5,
                        help="Define the number of epochs to use for the training process")
    parser.add_argument("-D", "--dim", dest="dim", type=int, help="Define the size of the vectors")
    args = parser.parse_args()

    server_flag = False
    if args.server_local.lower().strip() == 'server':
        server_flag = True

    if server_flag:
        path_prefix = "/home/findwise/interactionwise/wikipedia_dump/"
    else:
        path_prefix = "/media/azfar/New Volume/WikiDump/"

    vrd_embedder = VRDEmbedding(path_prefix, args.dim)

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
    glove = Glove(no_components=args.dim)
    print("Fitting Glove embeddings on corpus matrix...")
    glove.fit(corpus.matrix, epochs=args.num_epochs, no_threads=args.num_cores, verbose=True)
    print("Adding Glove dictionary...")
    glove.add_dictionary(corpus.dictionary)
    print("Saving Glove model to disk...")
    glove.save(os.path.join(path_prefix, "glove_epoch_5_dim_{}.model".format(args.dim)))

    print(glove.word_vectors[glove.dictionary['person']])
