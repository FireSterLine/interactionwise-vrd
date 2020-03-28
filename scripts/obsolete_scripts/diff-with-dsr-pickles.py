# FIX THE -1 thing
import os.path as osp
import pickle
import json
import sys

with open(osp.join("data", "vrd", "{}.pkl".format("test")), 'rb') as fid:
  data_pkl = pickle.load(fid, encoding="latin1")

for stage in ["test", "train"]:
  with open(osp.join("data", "vrd", "{}.pkl".format(stage)), 'rb') as fid:
    data_pkl = pickle.load(fid, encoding="latin1")
  #TEST prepare_data from dsr pickles with open("data/vrd/dsr_img_rels_{}.json".format(stage), 'rb') as f:
  #  margin=0
  with open("data/vrd/data_img_rels_{}.json".format(stage), 'rb') as f:
    data_new = json.load(f)
  assert len(data_pkl) == len(data_new), "Length is different: {} != {}".format(len(data_pkl), len(data_new))
  for i,(pkl,new) in enumerate(zip(data_pkl,data_new)):
    if pkl is None:
      print(i, " -> NONE")
      continue
    print(i)
    im_pkl = pkl["img_path"]
    im_new = new[0]
    assert im_pkl[-6:] == im_new[-6:], "Length is different: {} != {}".format(im_pkl, im_new)
    rels_new = new[1]
    print(pkl)
    print(new)
    print()
    pkl["boxes"], pkl["classes"]
    pkl["ix1"], pkl["ix2"], pkl["rel_classes"]
    for i_so,(ix1,ix2,predicates) in enumerate(zip(pkl["ix1"], pkl["ix2"], pkl["rel_classes"])):
      id_subject = pkl["classes"][ix1]
      id_object  = pkl["classes"][ix2]
      pkl_box1 = {"xmin" : pkl["boxes"][ix1][0], "ymin" : pkl["boxes"][ix1][1], "xmax" : pkl["boxes"][ix1][2], "ymax" : pkl["boxes"][ix1][3]}
      pkl_box2 = {"xmin" : pkl["boxes"][ix2][0], "ymin" : pkl["boxes"][ix2][1], "xmax" : pkl["boxes"][ix2][2], "ymax" : pkl["boxes"][ix2][3]}
      for pred in predicates:
        found = False
        print("Rel:")
        print(id_subject,id_object, pkl_box1, pkl_box2,pred)
        def same_box(box2,box1, margin=margin):
          return ((box2["xmin"]-box1["xmin"]) <= margin and box2["xmin"]>=box1["xmin"]) and \
                 ((box2["ymin"]-box1["ymin"]) <= margin and box2["ymin"]>=box1["ymin"]) and \
                 ((box2["ymax"]-box1["ymax"]) <= margin and box2["ymax"]>=box1["ymax"]) and \
                 ((box2["xmax"]-box1["xmax"]) <= margin and box2["xmax"]>=box1["xmax"]),
        for i_rel_new,rel_new in enumerate(rels_new):
          print("\t", rel_new["subject"]["id"],rel_new["object"]["id"],rel_new["predicate"]["id"], rel_new["subject"]["bbox"], rel_new["object"]["bbox"])
          if    rel_new["subject"]["id"]   == id_subject \
            and rel_new["object"]["id"]    == id_object \
            and rel_new["predicate"]["id"] == pred \
            and same_box(rel_new["subject"]["bbox"],pkl_box1) \
            and same_box(rel_new["object"]["bbox"],pkl_box2):
            # and rel_new["subject"]["bbox"] == pkl_box1 \
            # and rel_new["object"]["bbox"]  == pkl_box2:
              print(rel_new["subject"]["bbox"],pkl_box1)
              print(rel_new["object"]["bbox"],pkl_box2)
              del rels_new[i_rel_new]
              found = True
              break
        if not found == True:
          print("Warning: existing relationship in pickl not found!")
          input()
    if not len(rels_new) == 0:
      print("Warning: {} remaining relatinoships in new:\n{}".format(len(rels_new), repr(rels_new)))
      input()
