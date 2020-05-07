import re
import os
import sys
import time
import pickle
import argparse
import unicodedata
from glob import glob
from gensim.models import Word2Vec, FastText
from gensim.models.callbacks import CallbackAny2Vec
from gensim.test.utils import datapath, get_tmpfile
from gensim.corpora import WikiCorpus, MmCorpus

# TODO: avoid this global vrd_tokenize crapery
vrd_tokenize = None


class VRDEmbedding:
    def __init__(self, path_prefix, dim, model='word2vec'):
        global vrd_tokenize
        vrd_tokenize = self.vrd_tokenize
        self.PAT_ALPHABETIC = re.compile(r'(((?![\d])\w)+)', re.UNICODE)
        self.TOKEN_MIN_LEN = 2
        self.TOKEN_MAX_LEN = 15
        self.unicode = str
        self.path_prefix = path_prefix
        self.dim = dim
        self.model_name = model
        if model == 'word2vec':
            self.model = Word2Vec
        elif model == 'fasttext':
            self.model = FastText
        self.multi_word_phrases = ["traffic light", "trash can", "next to", "sleep next to", "sit next to",
                                   "stand next to", "park next", "walk next to", "stand behind", "sit behind",
                                   "park behind", "in the front of", "stand under", "sit under", "walk to", "walk past",
                                   "walk beside", "on the top of", "on the left of", "on the right of", "sit on",
                                   "stand on", "attach to", "adjacent to", "drive on", "taller than", "park on",
                                   "lying on", "lean on", "play with", "sleep on", "outside of", "rest on", "skate on",
                                   "banana bunch", "mountain range", "door frame", "tail fin", "telephone pole",
                                   "moustache", "train platform", "purple flower", "left ear", "tennis net",
                                   "windshield wiper", "bus stop", "lamp shade", "light switch", "shower curtain",
                                   "cardboard box", "table cloth", "doughnut", "laptop computer", "parking lot",
                                   "guard rail", "tv stand", "traffic signal", "tennis racket", "flower pot",
                                   "number 2", "baseball uniform", "fence post", "left hand", "palm tree",
                                   "ceiling fan", "clock hand", "lamp post", "light pole", "oven door", "traffic sign",
                                   "baseball cap", "tree top", "light bulb", "computer monitor", "door knob",
                                   "baseball field", "grass patch", "passenger car", "tennis ball", "window sill",
                                   "shower head", "name tag", "front window", "computer mouse", "cutting board",
                                   "hind leg", "paper towel", "computer screen", "tissue box", "american flag",
                                   "evergreen tree", "tree trunk", "mouse pad", "baseball glove", "minute hand",
                                   "window pane", "coffee maker", "front wheel", "road sign", "steering wheel",
                                   "tennis player", "manhole cover", "stop light", "street sign", "train station",
                                   "brake light", "wine glass"]

        self.epoch_logger = EpochLogger()
        self.epoch_saver = EpochSaver(path_prefix=path_prefix, dim=dim, model_name=model)

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

    def train_model(self, num_cores, num_epochs=None, model_file=None, serialize=False, server_flag=False):
        wiki_path = os.path.join(self.path_prefix, "wiki.pkl")
        if os.path.exists(wiki_path):
            print("Loading wiki from pickle file!")
            wiki = pickle.load(open(wiki_path, 'rb'))
            wiki.fname = os.path.join(path_prefix, "enwiki-latest-pages-articles.xml.bz2")
        else:
            print("Creating datapath...")
            # if it's the local environment, we are testing on the shortened wiki texts
            if server_flag is False:
                path_to_wiki_dump = datapath("enwiki-latest-pages-articles1.xml-p000000010p000030302-shortened.bz2")
            else:
                path_to_wiki_dump = datapath(os.path.join(path_prefix, "enwiki-latest-pages-articles.xml.bz2"))

            print("Initializing corpus...")
            start = time.time()
            wiki = WikiCorpus(path_to_wiki_dump,
                              tokenizer_func=self.vrd_tokenize)  # create word->word_id mapping, ~8h on full wiki
            end = time.time()
            print("Time taken to initialize corpus: {}".format(end - start))
            # this is to that the original wiki.pkl does not get overwritten by the one generated for shortened wiki
            if server_flag is True:
                print("Dumping wiki to disk...")
                pickle.dump(wiki, open(os.path.join(path_prefix, "wiki.pkl"), 'wb'))

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

        print("Training {} embeddings with dimensionality {}...".format(self.model_name, self.dim))
        start = time.time()
        if model_file is None:
            model = self.model(wiki_iterator, size=self.dim, workers=num_cores, min_count=1, iter=num_epochs,
                               callbacks=[self.epoch_logger, self.epoch_saver])
        else:
            # NOTE: This is model fine-tuning
            model = self.load_model(model_file)
            # NOTE: Existing models renaming should only happen for those models which were trained for a number of
            #   epochs between epochs_trained and (epochs_trained + num_epochs). However, this causes the existing
            #   model (which is to be fine-tuned) to be renamed too, essentially creating a copy that we don't need.
            #   Technically this is how it should be, and what's happening right now - however, when fine-tuning a
            #   Word2Vec model on Wiki further, we theoretically have no need left for the original model then.
            #   Need to change?
            epochs_trained = int(re.search(r'(?<=epoch_)\d+', model_file).group(0))
            model.callbacks[0].epoch = epochs_trained + 1
            model.callbacks[1].epoch += 1
            model.train(wiki_iterator, total_examples=model.corpus_count, epochs=num_epochs)
            # if models_renamed_flag is True:
            #     self._rename_existing_models(model.callbacks[1].path_prefix, total_epochs, dim, revert=True)
        end = time.time()
        print("Time taken to train the model: {}".format(end - start))
        return model

    def fine_tune_model_coco(self, path_to_model, model_type, tokenized_captions, num_epochs):
        # load the model
        model = self.load_model(path_to_model)
        model.callbacks[0] = self.epoch_logger
        model.callbacks[1] = self.epoch_saver

        # we save the COCO models in a subdirectory called "coco". If this subdirectory doesn't exist, create it
        if 'coco' not in model.callbacks[1].path_prefix:
            model_dir = os.path.dirname(path_to_model)
            coco_path = os.path.join(model_dir, 'coco')
            if not os.path.exists(coco_path):
                os.mkdir(coco_path)
            model.callbacks[1].path_prefix = os.path.join(model.callbacks[1].path_prefix, "coco/")

        # get dimension from path_to_model
        dim = re.search(r'(?<=_dim_)\d+', path_to_model).group(0)
        # get the number of epochs the current model is trained on
        epochs_trained = int(re.search(r'(?<=epoch_)\d+', path_to_model).group(0))
        # set epoch counters for the callbacks such that they continue from last epoch number of original model
        model.callbacks[0].epoch = epochs_trained + 1
        model.callbacks[1].epoch = epochs_trained + 1
        # get file extensions for vectors and trainables of the model of choice (FastText or Word2Vec)
        model_vectors_ext = model.callbacks[1].vectors_ext
        model_trainables_ext = model.callbacks[1].trainables_ext

        # rename all model files which were trained for lesser epochs than num_epochs so they don't get overwritten
        models_renamed_flag = self._rename_existing_models(model.callbacks[1].path_prefix, model_type,
                                                           epochs_trained + num_epochs, dim, model_vectors_ext,
                                                           model_trainables_ext, revert=False)

        # finetune the model on COCO captions
        model.train(tokenized_captions, total_examples=len(tokenized_captions), epochs=num_epochs)

        # if any model files were renamed before fine-tuning, rename them back to their original names
        if models_renamed_flag is True:
            self._rename_existing_models(model.callbacks[1].path_prefix, model_type, epochs_trained + num_epochs, dim,
                                         model_vectors_ext, model_trainables_ext, revert=True)
        return model

    def _rename_existing_models(self, path_to_model, model_type, num_epochs, dim, vectors_ext, trainables_ext, revert=False):
        """
            Before fine-tuning a model on COCO over num_epochs iterations, we need to check and see if any models
            already exist in the given directory which have been trained over lesser number of epochs. If so, we need
            to rename them first before initiating the fine-tuning, so that they don't get overwritten by the
            EpochSaver. Likewise, we need to rename back to their original names after the current model has completed
            fine-tuning.

            If the `revert` flag is False, we rename every existing model file which has been trained for epochs less
            than num_epochs to backup names.
            If the `revert` flag is True, we rename every model that was renamed earlier back to its original name.

            TODO: Technically, this function should also be called when training an embedding model from scratch.
              However, we don't need to train a Word2Vec of the same dimension size over different number of epochs, so
              perhaps it is not required at the moment.
        """

        # this flag lets us know if any models were renamed, so that we can later know if this function needs to be
        #   called again with revert=True
        models_renamed = False
        dir_name = os.path.dirname(path_to_model)
        # if revert is True, get all model files that were renamed to backup files
        if revert is True:
            model_names = glob(os.path.join(dir_name, "*.model_backup"))
        # if revert is False, get all model files
        else:
            model_names = glob(os.path.join(dir_name, "*.model"))

        for mdl_name in model_names:
            # this regex gets the number of epochs the current model was trained on. Note that the dimension must match
            m = re.search(r'(?<={}_epoch_)\d+(?=_dim_{})'.format(model_type, dim), mdl_name)
            if m:
                mdl_epoch = int(m.group(0))
                if mdl_epoch < num_epochs:
                    print("Model being renamed: {}".format(mdl_name))
                    if revert is True:
                        # source model files will contain '_backup', destination model files will not
                        current_mdl_name = mdl_name.rsplit('_backup')[0]
                        src_mdl_name = mdl_name
                        dest_mdl_name = current_mdl_name
                        src_vec_name = current_mdl_name + '.' + vectors_ext + "_backup"
                        dest_vec_name = current_mdl_name + '.' + vectors_ext
                        src_trainables_name = current_mdl_name + '.' + trainables_ext + "_backup"
                        dest_trainables_name = current_mdl_name + '.' + trainables_ext
                    else:
                        # source model files will not contain '_backup', destination model files will
                        current_mdl_name = mdl_name
                        src_mdl_name = current_mdl_name
                        dest_mdl_name = current_mdl_name + "_backup"
                        src_vec_name = current_mdl_name + '.' + vectors_ext
                        dest_vec_name = current_mdl_name + '.' + vectors_ext + "_backup"
                        src_trainables_name = current_mdl_name + '.' + trainables_ext
                        dest_trainables_name = current_mdl_name + '.' + trainables_ext + "_backup"

                    os.rename(src_mdl_name, dest_mdl_name)
                    if os.path.exists(src_vec_name):
                        os.rename(src_vec_name, dest_vec_name)
                    if os.path.exists(src_trainables_name):
                        os.rename(src_trainables_name, dest_trainables_name)

                    models_renamed = True
        return models_renamed

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

    def __init__(self, path_prefix, dim, model_name):
        self.path_prefix = path_prefix
        self.dim = dim
        self.epoch = 1
        self.model_name = model_name
        if model_name == 'fasttext':
            self.vectors_ext = "wv.vectors_ngrams.npy"
            self.trainables_ext = "trainables.vectors_ngrams_lockf.npy"
        elif model_name == 'word2vec':
            self.vectors_ext = "wv.vectors.npy"
            self.trainables_ext = "trainables.syn1neg.npy"

    def on_epoch_end(self, model):
        # remove previously saved checkpoint for storage saving purposes
        prev_checkpoint = os.path.join(self.path_prefix, "{}_epoch_{}_dim_{}.model".format(
            self.model_name, self.epoch - 1, self.dim))
        prev_checkpoint_vectors = os.path.join(self.path_prefix, "{}_epoch_{}_dim_{}.model.{}".format(
            self.model_name, self.epoch - 1, self.dim, self.vectors_ext))
        prev_checkpoint_trainable = os.path.join(self.path_prefix, "{}_epoch_{}_dim_{}.model.{}".format(
            self.model_name, self.epoch - 1, self.dim, self.trainables_ext))

        if os.path.exists(prev_checkpoint):
            print("Removing previous checkpoint...")
            os.remove(prev_checkpoint)
            if os.path.exists(prev_checkpoint_vectors):
                os.remove(prev_checkpoint_vectors)
            if os.path.exists(prev_checkpoint_trainable):
                os.remove(prev_checkpoint_trainable)

        # save current epoch checkpoint to disk
        print("Saving checkpoint {}".format(self.epoch))
        output_path = get_tmpfile(os.path.join(self.path_prefix, "{}_epoch_{}_dim_{}.model".format(self.model_name,
                                                                                                   self.epoch,
                                                                                                   self.dim)))
        model.save(output_path)
        self.epoch += 1


class EpochLogger(CallbackAny2Vec):
    """
      This is a class for printing the start and end of each epoch, to monitor the model's training progress.
    """

    def __init__(self):
        self.epoch = 1

    def on_epoch_begin(self, model):
        print("Starting epoch # {}".format(self.epoch))

    def on_epoch_end(self, model):
        print("Ending epoch {}".format(self.epoch))
        print("----------------------")
        self.epoch += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a Word2Vec/FastText model on the Wikipedia dataset")
    parser.add_argument("-SL", "--server_local", dest="server_local", type=str, default="local",
                        help="Define whether the execution environment is the server or local")
    parser.add_argument("-C", "--num_cores", dest="num_cores", type=int, default=6,
                        help="Define the number of cores to use for the training process")
    parser.add_argument("-E", "--num_epochs", dest="num_epochs", type=int, default=5,
                        help="Define the number of epochs to use for the training process")
    parser.add_argument("-D", "--dim", dest="dim", type=int, help="Define the size of the vectors")
    parser.add_argument("-M", "--model", dest="model", type=str, default="word2vec",
                        help="Define whether to use Word2Vec or FastText for generating the embeddings")
    parser.add_argument("-F", "--model_file_name", dest="model_file_name", type=str, default=None,
                        help="Define the path to the model to finetune")
    server_flag = False
    serialize = False
    args = parser.parse_args()

    if args.server_local.lower().strip() == 'server':
        server_flag = True

    if server_flag:
        path_prefix = "/home/findwise/interactionwise/wikipedia_dump/"
    else:
        path_prefix = "/media/azfar/New Volume/WikiDump/"

    if args.model_file_name is not None:
        # we expect the models to be located in the path_prefix directory
        if "/" not in args.model_file_name:
            args.model_file_name = os.path.join(path_prefix, args.model_file_name)
        if not os.path.exists(args.model_file_name):
            print("The specified model '{}' could not be found on disk; please verify!".format(args.model_file_name))
            exit(1)

    vrd_embedder = VRDEmbedding(path_prefix, args.dim, args.model.lower())
    model = vrd_embedder.train_model(num_cores=args.num_cores, num_epochs=args.num_epochs, serialize=False,
                                     server_flag=server_flag, model_file=args.model_file_name)
    # fine_tuned_model = vrd_embedder.fine_tune_model_coco(
    #     os.path.join(path_prefix, "word2vec_epoch_5_dim_50.model"),
    #     model_type=args.model,
    #     tokenized_captions=pickle.load(open("../../coco_captions_tokenized.pkl", 'rb')),
    #     num_epochs=5
    # )

    # model = VRDEmbedding.load_model(os.path.join(path_prefix, "epoch_4_dim_50.model"))

    print("Wiki person: {}".format(model.wv['person']))
    # print("COCO person: {}".format(fine_tuned_model['person']))
