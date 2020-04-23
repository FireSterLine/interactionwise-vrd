import os
import re
import json
import spacy
import pickle
from collections import defaultdict


def lemmatize_captions(captions, predicate_verbs, nlp):
    """
        This function lemmatizes the words in each caption, and then bundles them back into a string.
        TODO: Do we also want to deal with lemmatized objects?
    """
    lemma_captions = []
    for ix, caption in enumerate(captions):
        if ix % 5000 == 0 and ix > 0:
            print("\t{} captions lemmatized!".format(ix))
        lemma_words = []
        doc = nlp(caption)
        for word in doc:
            # only add the lemmatized word if it exists in the lemmatized predicate verbs. Otherwise, add original word
            if word.lemma_.lower() in predicate_verbs:
                # we do not add the lowercased term here, since the model can take care of that itself
                lemma_words.append(word.lemma_)
            else:
                # we do not add the lowercased term here, since the model can take care of that itself
                lemma_words.append(word.text)
        lemma_captions.append(" ".join(lemma_words))
    return lemma_captions


def vrd_tokenize(captions, multi_word_phrases, multi_word_phrase_mappings, tok_rg):
    """
        This function first combines all instances of multi-word phrases in each caption via underscore, and then
        tokenizes each caption using the same tokenization regex used while training Word2Vec over Wikipedia texts.
    """
    tok_captions = []
    for caption in captions:
        # first, replace all terms in multi-word phrase mappings with corresponding standardized terms
        # so for example, "in front of" becomes "in the front of"
        for mwp_k, mwp_v in multi_word_phrase_mappings.items():
            caption = re.sub(r'\b%s\b' % mwp_k, mwp_v, caption)
        # next, look for multi-word phrases in this caption, and join them via underscore
        for term in multi_word_phrases:
            replace_term = '_'.join(term.split())
            # replace the multi-word phrase by replace_term
            caption = re.sub(r'\b%s\b' % term, replace_term, caption)
            # try:
            #     # if this multi-word phrase exists in multi-word mappings, use the standard value for replacement
            #     std_replace_term = multi_word_phrase_mappings[replace_term]
            #     # replace the replace_term with std_replace term in caption
            #     caption = re.sub(r'\b%s\b' % replace_term, std_replace_term, caption)
            # except KeyError:
            #     pass
        # tok_caption = caption
        tok_caption = []
        # this regex does the tokenization
        matches = tok_rg.finditer(caption)
        for m in matches:
            tok_caption.append(m.group())
        tok_captions.append(tok_caption)
    return tok_captions


def explore_captions_data(tok_captions, obj_filename, pred_filename):
    objects = json.load(open(obj_filename, 'r'))
    predicates = json.load(open(pred_filename, 'r'))
    # combine objects and predicates, join multi-word elements with underscore
    combined_elems = ['_'.join(a.split()) for a in objects + predicates]
    # dictionary to maintain counts of objects and predicates found in tokenized COCO captions
    obj_pred_count = dict.fromkeys(combined_elems, 0)
    for tok_cap in tok_captions:
        for elem in combined_elems:
            if elem in tok_cap:
                obj_pred_count[elem] += 1
    return obj_pred_count


if __name__ == '__main__':
    tokenize_regex = re.compile(r'(((?![\d])\w)+)', re.UNICODE)
    captions_filename = "../../coco_captions.txt"
    vrd_objects_filename = "../data/vrd/objects.json"
    vrd_predicates_filename = "../data/vrd/predicates.json"
    vrd_predicate_verbs_filename = "../data/vrd/predicate_verbs.json"
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

    multi_word_phrase_mappings = {
        "in front of": "in the front of",
        "on top of": "on the top of",
        # these two keys below don't exist in COCO
        # "on_left_of": "on_the_left_of",        # maybe "on left of" as replacement value can be changed to "to left of"?
        # "on_right_of": "on_the_right_of",
        # self-introduced ones
        "to the left of": "on the left of",
        "to the right of": "on the right of",
        "to left of": "on the left of",
        "to right of": "on the right of"
    }

    # output filenames
    tokenized_captions_pickle = "../../coco_captions_tokenized.pkl"
    tokenized_captions_txt = "../../coco_captions_tokenized.txt"

    if not os.path.exists(tokenized_captions_pickle) and not os.path.exists(tokenized_captions_txt):
        # read captions from file
        captions = []
        print("Loading captions...")
        with open(captions_filename, 'r') as rfile:
            for line in rfile:
                captions.append(line.strip())

        # load Spacy English model - this will be used for lemmatization
        nlp = spacy.load('en_core_web_sm')
        # get lemmatized captions
        print("Lemmatizing captions...")
        predicate_verbs = json.load(open(vrd_predicate_verbs_filename, 'r'))
        lemma_captions = lemmatize_captions(captions, predicate_verbs, nlp)
        # get captions with multi-word phrases joined by underscore
        print("Tokenizing captions...")
        tokenized_captions = vrd_tokenize(lemma_captions, multi_word_phrases, multi_word_phrase_mappings, tokenize_regex)
        for tok_caption in tokenized_captions:
            print(tok_caption)

        print("Writing tokenized captions to file...")
        with open(tokenized_captions_pickle, 'wb') as wfile:
            pickle.dump(tokenized_captions, wfile)

        with open(tokenized_captions_txt, 'w') as wfile:
            for tok_caption in tokenized_captions:
                wfile.write(' '.join(tok_caption))
                wfile.write('\n')
    else:
        print("Tokenized captions have already been generated!")
        tokenized_captions = pickle.load(open(tokenized_captions_pickle, 'rb'))

    print("Exploring data...")
    vrd_elem_counts = explore_captions_data(tokenized_captions, vrd_objects_filename, vrd_predicates_filename)
    for k, v in vrd_elem_counts.items():
        print("{}: {}".format(k, v))
