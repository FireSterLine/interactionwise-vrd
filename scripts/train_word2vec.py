import re
import os
import sys
import time
import pickle
import unicodedata
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
from gensim.test.utils import datapath, get_tmpfile
from gensim.corpora import WikiCorpus, MmCorpus

# TODO: avoid this global vrd_tokenize crapery
vrd_tokenize = None 

class VRDEmbedding:
    def __init__(self, path_prefix, dim):
        global vrd_tokenize
        vrd_tokenize = self.vrd_tokenize
        self.PAT_ALPHABETIC = re.compile(r'(((?![\d])\w)+)', re.UNICODE)
        self.TOKEN_MIN_LEN = 2
        self.TOKEN_MAX_LEN = 15
        self.unicode = str
        self.path_prefix = path_prefix
        self.dim = dim
        self.multi_word_phrases = ["traffic light", "trash can", "next to", "sleep next to", "sit next to", "stand next to", "park next", "walk next to", "stand behind", "sit behind", "park behind", "in the front of", "stand under", "sit under", "walk to", "walk past", "walk beside", "on the top of", "on the left of", "on the right of", "sit on", "stand on", "attach to", "adjacent to", "drive on", "taller than", "park on", "lying on", "lean on", "play with", "sleep on", "outside of", "rest on", "skate on", "banana bunch", "mountain range", "door frame", "tail fin", "telephone pole", "moustache", "train platform", "purple flower", "left ear", "tennis net", "windshield wiper", "bus stop", "lamp shade", "light switch", "shower curtain", "cardboard box", "table cloth", "doughnut", "laptop computer", "parking lot", "guard rail", "tv stand", "traffic signal", "tennis racket", "flower pot", "number 2", "baseball uniform", "fence post", "left hand", "palm tree", "ceiling fan", "clock hand", "lamp post", "light pole", "oven door", "traffic sign", "baseball cap", "tree top", "light bulb", "computer monitor", "door knob", "baseball field", "grass patch", "passenger car", "tennis ball", "window sill", "shower head", "name tag", "front window", "computer mouse", "cutting board", "hind leg", "paper towel", "computer screen", "tissue box", "american flag", "evergreen tree", "tree trunk", "mouse pad", "baseball glove", "minute hand", "window pane", "coffee maker", "front wheel", "road sign", "steering wheel", "tennis player", "manhole cover", "stop light", "street sign", "train station", "brake light", "wine glass"]

        self.epoch_logger = EpochLogger()
        self.epoch_saver = EpochSaver(path_prefix=path_prefix, dim=dim)

    def _tokenize(self, text, lowercase=False, deacc=False):
        if lowercase:
            text = text.lower()
        if deacc:
            text = self._deaccent(text)
        return self._simple_tokenize(text)

    def _simple_tokenize(self, text):
        for match in self.PAT_ALPHABETIC.finditer(text):
            yield match.group()

    def _deaccent(self, text):
        norm = unicodedata.normalize("NFD", text)
        result = ''.join(ch for ch in norm if unicodedata.category(ch) != 'Mn')
        return unicodedata.normalize("NFC", result)

    def vrd_tokenize(self, content, token_min_len=2, token_max_len=15, lowercase=True):
        for word in self.multi_word_phrases:
            content = re.sub(r'\b%s\b' % word, '_'.join(word.split()), content)
        return [
            token for token in self._tokenize(content, lowercase=lowercase)
            if token_min_len <= len(token) <= token_max_len and not token.startswith('_')
        ]

    def train_model(self, num_cores, serialize=False):
        wiki_path = os.path.join(self.path_prefix, "wiki.pkl")
        if os.path.exists(wiki_path):
            print("Loading wiki from pickle file!")
            wiki = pickle.load(open(wiki_path, 'rb'))
            wiki.fname = os.path.join(path_prefix, "enwiki-latest-pages-articles.xml.bz2")
        else:
            print("Creating datapath...")
            # path_to_wiki_dump = datapath("enwiki-latest-pages-articles1.xml-p000000010p000030302-shortened.bz2")
            path_to_wiki_dump = datapath(os.path.join(path_prefix, "enwiki-latest-pages-articles.xml.bz2"))

            print("Initializing corpus...")
            start = time.time()
            wiki = WikiCorpus(path_to_wiki_dump,
                              tokenizer_func=self.vrd_tokenize)  # create word->word_id mapping, ~8h on full wiki
            end = time.time()
            print("Time taken to initialize corpus: {}".format(end - start))
            print("Dumping wiki to disk...")
            pickle.dump(wiki, open(os.path.join(path_prefix, "wiki.pkl"), 'w'))

            if serialize is True:
                print("Creating corpus path...")
                corpus_path = get_tmpfile(os.path.join(path_prefix, "wiki-corpus.mm"))
                print("Serializing corpus...")
                start = time.time()
                MmCorpus.serialize(corpus_path, wiki)
                end = time.time()
                print("Time taken to serialize corpus: {}".format(end - start))

        # make an iterator around the get_texts() function of wiki
        wiki_iterator = self.MakeIter(wiki)

        print("Training Word2Vec with dimensionality {}...".format(dim))
        start = time.time()
        model = Word2Vec(wiki_iterator, size=self.dim, workers=num_cores, min_count=1, callbacks=[
            self.epoch_logger, self.epoch_saver])
        end = time.time()
        print("Time taken to train the model: {}".format(end - start))
        return model

    @staticmethod
    def load_model(model_path):
        return Word2Vec.load(model_path)

    class MakeIter(object):
        """
            We are wrapping the wiki.get_texts() function (which returns a generator) inside an Iterator,
            since Word2Vec requires an iterable for training.
        """
        def __init__(self, wiki):
            self.wiki = wiki

        def __iter__(self):
            return self.wiki.get_texts()

class EpochSaver(CallbackAny2Vec):
        """
            This is a class for saving the model checkpoint on each epoch, and deleting the previous checkpoint.
        """
        def __init__(self, path_prefix, dim):
            self.path_prefix = path_prefix
            self.dim = dim
            self.epoch = 0

        def on_epoch_end(self, model):
            print("Saving checkpoint {}".format(self.epoch))
            output_path = get_tmpfile(os.path.join(self.path_prefix, "epoch_{}_dim_{}.model".format(self.epoch, self.dim)))
            model.save(output_path)
            # remove previously saved checkpoint for storage saving purposes
            prev_checkpoint = os.path.join(self.path_prefix, "epoch_{}_dim_{}.model".format(self.epoch - 1, self.dim))
            prev_checkpoint_vectors = os.path.join(self.path_prefix, "epoch_{}_dim_{}.model.wv.vectors.npy".format(self.epoch - 1, self.dim))
            prev_checkpoint_trainable = os.path.join(self.path_prefix, "epoch_{}_dim_{}.trainables.syn1neg.npy".format(self.epoch - 1, self.dim))
            if os.path.exists(prev_checkpoint):
                print("Removing previous checkpoint...")
                os.remove(prev_checkpoint)
                if os.path.exists(prev_checkpoint_vectors):
                    os.remove(prev_checkpoint_vectors)
                if os.path.exists(prev_checkpoint_trainable):
                    os.remove(prev_checkpoint_trainable)
            self.epoch += 1

class EpochLogger(CallbackAny2Vec):
        """
            This is a class for printing the start and end of each epoch, to monitor the model's training progress.
        """
        def __init__(self):
            self.epoch = 0

        def on_epoch_begin(self, model):
            print("Starting epoch # {}".format(self.epoch))

        def on_epoch_end(self, model):
            print("Ending epoch {}".format(self.epoch))
            print("----------------------")
            self.epoch += 1


if __name__ == '__main__':
    server_local = sys.argv[1]
    num_cores = int(sys.argv[2])
    dim = int(sys.argv[3])
    server_flag = False
    serialize = False

    if server_local.lower().strip() == 'server':
        server_flag = True

    if server_flag:
        path_prefix = "/home/findwise/interactionwise/wikipedia_dump/"
    else:
        path_prefix = "/media/azfar/New Volume/WikiDump/"

    vrd_embedder = VRDEmbedding(path_prefix, dim)
    model = vrd_embedder.train_model(num_cores=num_cores, serialize=False)

    #model = VRDEmbedding.load_model(os.path.join(path_prefix, "epoch_4_dim_50.model"))
    #model = VRDEmbedding.load_model(os.path.join(path_prefix, "epoch_4_dim_100.model"))

    # print("Dumping model to disk...")
    # model.save("/media/azfar/New Volume/WikiDump/word2vec_model")

    print("person: ")
    print(model.wv['person'])
