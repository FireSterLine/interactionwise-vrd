#!/usr/bin/python


''' Visual genome data analysis and preprocessing.'''

import os
import json
import operator
from pprint import pprint
from shutil import copyfile
from collections import Counter
from nltk.stem import WordNetLemmatizer
import xml.etree.cElementTree as ET
from xml.dom import minidom

# Set maximum values for number of object / attribute / relation classes,
# Changing the number of objects will require retraining the object detection model
# max_objects, max_attributes, max_relations = (2500, 1000, 500)
max_objects, max_attributes, max_relations = (1600, 400, 20)
# max_objects, max_attributes, max_relations = (150, 50, 50)

# TODO for Azfar: move your local vg directory to data/vg so there is no "local/server difference"
dataDir = os.path.join("data", "vg")
outDir = os.path.join("data", "genome", "{}-{}-{}".format(max_objects, max_attributes, max_relations))
if not os.path.exists(outDir):
    os.mkdir(outDir)

common_attributes = set(['white', 'black', 'blue', 'green', 'red', 'brown', 'yellow',
                         'small', 'large', 'silver', 'wooden', 'orange', 'gray', 'grey', 'metal', 'pink', 'tall',
                         'long', 'dark'])
lemmatizer = WordNetLemmatizer()
output_format = 'json'
lemma_flag = True


def clean_string(string, pos_tag):
    ''' Convert to lowercase, strip, lemmatize, remove trailing full stop '''
    string = string.lower().strip()
    string = string.replace("-", " ")
    if lemma_flag is True:
        words = string.split()
        string = " ".join([lemmatizer.lemmatize(w, pos_tag) for w in words])
    if len(string) >= 1 and string[-1] == '.':
        return string[:-1].strip()
    return string


def clean_objects(string, common_attributes):
    ''' Return object and attribute lists '''
    string = clean_string(string, 'n')
    words = string.split()
    if len(words) > 1:
        prefix_words_are_adj = True
        for att in words[:-1]:
            if att not in common_attributes:
                prefix_words_are_adj = False
        if prefix_words_are_adj:
            return words[-1:], words[:-1]
        else:
            return [string], []
    else:
        return [string], []


def clean_attributes(string):
    ''' Return attribute list '''
    string = clean_string(string, 'n')
    if string == "black and white":
        return [string]
    else:
        return [word.lower().strip() for word in string.split(" and ")]


def clean_relations(string):
    ''' Apply string cleaning operation on predicate '''
    string = clean_string(string, 'v')
    if len(string) > 1:
        return [string]
    else:
        return []


def build_vocabs_and_json():
    objects = Counter()
    attributes = Counter()
    relations = Counter()

    print("Loading data from scene graphs...")
    with open(os.path.join(dataDir, 'scene_graphs.json')) as f:
        data = json.load(f)

    print("Lemmatizing set to {}".format(lemma_flag))
    print("Extracting and cleaning attributes and predicates...")
    # First extract attributes and relations
    for sg in data:
        # get attributes from all objects associated with this element
        attrs = [x['attributes'] for x in sg['objects'] if 'attributes' in x.keys()]
        attrs = sum(attrs, [])
        for attr in attributes:
            try:
                attributes.update(clean_attributes(attr))
            except:
                pass
        for rel in sg['relationships']:
            relations.update(clean_relations(rel['predicate']))

    print("Extracting and cleaning objects...")
    # Now extract objects, while looking for common adjectives that will be repurposed
    # as attributes
    for sg in data:
        for obj in sg['objects']:
            o, a = clean_objects(obj['names'][0], common_attributes)
            objects.update(o)
            attributes.update(a)

    print("Writing full objects, attributes and predicates to files...")
    with open(os.path.join(outDir, "objects_count.txt"), "w") as text_file:
        for k, v in sorted(objects.items(), key=operator.itemgetter(1), reverse=True):
            text_file.write("%s\t%d\n" % (k.encode('utf-8'), v))

    with open(os.path.join(outDir, "attributes_count.txt"), "w") as text_file:
        for k, v in sorted(attributes.items(), key=operator.itemgetter(1), reverse=True):
            text_file.write("%s\t%d\n" % (k.encode('utf-8'), v))

    with open(os.path.join(outDir, "relations_count.txt"), "w") as text_file:
        for k, v in sorted(relations.items(), key=operator.itemgetter(1), reverse=True):
            text_file.write("%s\t%d\n" % (k.encode('utf-8'), v))

    # Create full-sized vocabs
    print("Creating vocabularies...")
    print("Most common objects: ")
    pprint(objects.most_common(10))
    print("Most common predicates:")
    pprint(relations.most_common(10))

    objects = set([k for k, v in objects.most_common(max_objects)])
    attributes = set([k for k, v in attributes.most_common(max_attributes)])
    relations = set([k for k, v in relations.most_common(max_relations)])

    print("Writing condensed objects, attributes and predicates to files...")
    with open(os.path.join(outDir, "objects_vocab.txt"), "w") as text_file:
        for item in objects:
            text_file.write("%s\n" % item)
    with open(os.path.join(outDir, "attributes_vocab.txt"), "w") as text_file:
        for item in attributes:
            text_file.write("%s\n" % item)
    with open(os.path.join(outDir, "relations_vocab.txt"), "w") as text_file:
        for item in relations:
            text_file.write("%s\n" % item)

    print("Generating {} output...".format(output_format.upper()))
    # Load image metadata
    metadata = {}
    with open(os.path.join(dataDir, 'image_data.json')) as f:
        for item in json.load(f):
            metadata[item['image_id']] = item

    # Output clean json files, one per image
    out_folder = output_format
    if not os.path.exists(os.path.join(outDir, out_folder)):
        os.mkdir(os.path.join(outDir, out_folder))

    for index, sg in enumerate(data):
        # if index >= 500:
        #     break
        meta = metadata[sg["image_id"]]
        assert sg["image_id"] == meta["image_id"]
        url_split = meta["url"].split("/")
        if output_format == 'json':
            ann = {}
            ann['folder'] = url_split[-2]
            ann['filename'] = url_split[-1]

            ann['source'] = {}
            ann['source']['database'] = "Visual Genome Version 1.4"
            ann['source']['image_id'] = str(meta['image_id'])
            ann['source']['coco_id'] = str(meta['coco_id'])
            ann['source']['flickr_id'] = str(meta['flickr_id'])

            ann['size'] = {}
            ann['size']['width'] = str(meta['width'])
            ann['size']['height'] = str(meta['height'])
            ann['size']['depth'] = "3"

            ann['segmented'] = "0"

        elif output_format == 'xml':
            ann = ET.Element("annotation")
            ET.SubElement(ann, "folder").text = url_split[-2]
            ET.SubElement(ann, "filename").text = url_split[-1]

            source = ET.SubElement(ann, "source")
            ET.SubElement(source, "database").text = "Visual Genome Version 1.4"
            ET.SubElement(source, "image_id").text = str(meta["image_id"])
            ET.SubElement(source, "coco_id").text = str(meta["coco_id"])
            ET.SubElement(source, "flickr_id").text = str(meta["flickr_id"])

            size = ET.SubElement(ann, "size")
            ET.SubElement(size, "width").text = str(meta["width"])
            ET.SubElement(size, "height").text = str(meta["height"])
            ET.SubElement(size, "depth").text = "3"

            ET.SubElement(ann, "segmented").text = "0"

        object_set = set()
        if output_format == 'json':
            ann['objects'] = []
        for obj in sg['objects']:
            o, a = clean_objects(obj['names'][0], common_attributes)
            if o[0] in objects:
                if output_format == 'json':
                    ob = {}
                    ob['name'] = o
                    ob['object_id'] = str(obj['object_id'])
                    object_set.add(obj['object_id'])
                    ob['difficult'] = "0"

                    ob['bndbox'] = {}
                    ob['bndbox']['xmin'] = str(obj['x'])
                    ob['bndbox']['ymin'] = str(obj['y'])
                    ob['bndbox']['xmax'] = str(obj['x'] + obj['w'])
                    ob['bndbox']['ymax'] = str(obj['y'] + obj['h'])
                    ann['objects'].append(ob)
                elif output_format == 'xml':
                    ob = ET.SubElement(ann, "object")
                    ET.SubElement(ob, "name").text = o[0]
                    ET.SubElement(ob, "object_id").text = str(obj["object_id"])
                    object_set.add(obj["object_id"])
                    ET.SubElement(ob, "difficult").text = "0"

                    bbox = ET.SubElement(ob, "bndbox")
                    ET.SubElement(bbox, "xmin").text = str(obj["x"])
                    ET.SubElement(bbox, "ymin").text = str(obj["y"])
                    ET.SubElement(bbox, "xmax").text = str(obj["x"] + obj["w"])
                    ET.SubElement(bbox, "ymax").text = str(obj["y"] + obj["h"])

        if output_format == 'json':
            ann['relations'] = []

        for rel in sg['relationships']:
            predicate = clean_string(rel["predicate"], 'v')
            if rel["subject_id"] in object_set and rel["object_id"] in object_set:
                if predicate in relations:
                    if output_format == 'json':
                        rel_ = {}
                        rel_['subject_id'] = str(rel['subject_id'])
                        rel_['object_id'] = str(rel['object_id'])
                        rel_['predicate'] = predicate
                        ann['relations'].append(rel_)
                    elif output_format == 'xml':
                        re = ET.SubElement(ann, "relation")
                        ET.SubElement(re, "subject_id").text = str(rel["subject_id"])
                        ET.SubElement(re, "object_id").text = str(rel["object_id"])
                        ET.SubElement(re, "predicate").text = predicate

        outFile = url_split[-1].replace(".jpg", ".json")

        # write JSON/XML to file
        if output_format == 'json':
            if len(ann['objects']) > 0:
                json.dump(ann, open(os.path.join(outDir, out_folder, outFile), 'w'))
        elif output_format == 'xml':
            tree = ET.ElementTree(ann)
            if len(tree.findall('object')) > 0:
                tree.write(os.path.join(outDir, out_folder, outFile))


if __name__ == "__main__":

    # UPDATE: We don't need this anymore since we already have scene_graphs.json in the dir
    # First, use visual genome library to merge attributes and scene graphs
    # vg.AddAttrsToSceneGraphs(dataDir=dataDir)

    # Next, build json files
    build_vocabs_and_json()
