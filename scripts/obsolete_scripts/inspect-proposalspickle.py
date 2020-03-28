import pickle
import scipy.io as sio
import numpy as np

det_result = sio.loadmat("data/vrd/eval/from-language-priors/det_result.mat")

# Create pickle from matlab object
new_proposal_pkl = {}
new_proposal_pkl["boxes"] = []
new_proposal_pkl["cls"]   = []
new_proposal_pkl["confs"] = []

# Check the pickle provided by DSR
with open("data/vrd/eval/det_res.pkl.dsr", 'rb') as f:
  proposal_pkl = pickle.load(f, encoding="latin1")

for i,a in enumerate(proposal_pkl["boxes"]):
  for x,y in enumerate(a):
    if len(y) != 4:
      print("!")
      print(i,x)

for i,a in enumerate(proposal_pkl["cls"]):
  for x,y in enumerate(a):
    if len(y) != 1:
      print("!")
      print(i,x)


for i,a in enumerate(proposal_pkl["confs"]):
  for x,y in enumerate(a):
    if len(y) != 1:
      print("!")
      print(i,x)

print(det_result.keys())
print(proposal_pkl.keys())

assert len(det_result["detection_bboxes"]) == 1, "ERROR. Malformed .mat file"
assert len(det_result["detection_labels"]) == 1, "ERROR. Malformed .mat file"
assert len(det_result["detection_confs"])  == 1, "ERROR. Malformed .mat file"

print(len(det_result["detection_bboxes"][0]))
print(len(det_result["detection_labels"][0]))
print(len(det_result["detection_confs"][0]))

print(len(proposal_pkl["boxes"]))
print(len(proposal_pkl["cls"]))
print(len(proposal_pkl["confs"]))

print()

for i,(lp_boxes, lp_cls, lp_confs, boxes, cls, confs) in \
        enumerate(zip(det_result["detection_bboxes"][0],
                      det_result["detection_labels"][0],
                      det_result["detection_confs"][0],
                      proposal_pkl["boxes"],
                      proposal_pkl["cls"],
                      proposal_pkl["confs"])
      ):
        print("{}:".format(i))
        print((lp_boxes))
        print((lp_cls))
        print((lp_confs))
        print((boxes))
        print((cls))
        print((confs))

        transf_lp_boxes = lp_boxes-1
        transf_lp_cls   = lp_cls-1
        transf_lp_confs = lp_confs

        new_proposal_pkl["boxes"].append(np.array(transf_lp_boxes, dtype=np.int))
        new_proposal_pkl["cls"]  .append(np.array(transf_lp_cls,   dtype=np.int))
        new_proposal_pkl["confs"].append(np.array(transf_lp_confs, dtype=np.float32))

        proposal_pkl["boxes"][i] = np.array(proposal_pkl["boxes"][i], dtype=np.int)
        proposal_pkl["cls"][i]   = np.array(proposal_pkl["cls"][i],   dtype=np.int)
        proposal_pkl["confs"][i] = np.array(proposal_pkl["confs"][i], dtype=np.float32)

        if lp_boxes.size == 0:
          if boxes.size != 0:
            print("non-empty + empty!")
            input()

          continue

        if np.max(np.abs(transf_lp_boxes-boxes)) != 0:
          print("boxes:")
          print(lp_boxes, boxes)
          input()

        if np.max(np.abs(transf_lp_cls-cls)) != 0:
          print("cls:")
          print(lp_cls, cls)
          input()

        if np.max(np.abs(transf_lp_confs-confs)) > 0.0001:
          print("confs:")
          print(lp_confs, confs)
          input()


# new_proposal_pkl["boxes"] = np.array(new_proposal_pkl["boxes"], dtype=np.int)
# new_proposal_pkl["cls"]   = np.array(new_proposal_pkl["cls"],   dtype=np.int)
# new_proposal_pkl["confs"] = np.array(new_proposal_pkl["confs"], dtype=np.float32)

# CHECK:
print()

# print("new_proposal_pkl == proposal_pkl: ", new_proposal_pkl == proposal_pkl)
# print(np.array_equiv(new_proposal_pkl, proposal_pkl))
# print(np.array_equal(new_proposal_pkl, proposal_pkl))

for i,(new_boxes, new_cls, new_confs, boxes, cls, confs) in \
        enumerate(zip(new_proposal_pkl["boxes"],
                      new_proposal_pkl["cls"],
                      new_proposal_pkl["confs"],
                      proposal_pkl["boxes"],
                      proposal_pkl["cls"],
                      proposal_pkl["confs"])
      ):
      print(np.array_equiv(new_boxes, boxes))
      print(np.array_equiv(new_cls, cls))
      print(np.array_equiv(new_confs, confs))
      print(np.allclose(new_boxes, boxes))
      print(np.allclose(new_cls, cls))
      print(np.allclose(new_confs, confs))
      if not (np.array_equiv(new_boxes, boxes) and \
          np.array_equiv(new_cls, cls) and \
          np.array_equiv(new_confs, confs) and \
          np.allclose(new_boxes, boxes) and \
          np.allclose(new_cls, cls) and \
          np.allclose(new_confs, confs)):
            input("ERROR! pickles are not the same")

if not (len(new_proposal_pkl["boxes"]) == len(new_proposal_pkl["cls"]) == len(new_proposal_pkl["confs"]) == \
            len(proposal_pkl["boxes"]) ==     len(proposal_pkl["cls"])     == len(proposal_pkl["confs"])):
        input("ERROR! pickles are not the same")

# print(np.allclose(new_proposal_pkl, new_proposal_pkl))
# print(np.allclose(proposal_pkl, proposal_pkl))
# print(np.allclose(new_proposal_pkl, list(proposal_pkl)))
# print(np.allclose(new_proposal_pkl, list(proposal_pkl), rtol=.001, atol=.001))

with open("data/vrd/eval/det_res.pkl", 'wb') as f:
  pickle.dump(new_proposal_pkl, f)
