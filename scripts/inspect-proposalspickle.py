import pickle
import scipy.io as sio
import numpy as np

det_result = sio.loadmat("data/vrd/eval/from-language-prior/det_result.mat")

with open("data/vrd/eval/det_res.pkl", 'rb') as f:
  proposal = pickle.load(f, encoding="latin1")

for i,a in enumerate(proposal["boxes"]):
  for x,y in enumerate(a):
    if len(y) != 4:
      print("!")
      print(i,x)

for i,a in enumerate(proposal["cls"]):
  for x,y in enumerate(a):
    if len(y) != 1:
      print("!")
      print(i,x)


for i,a in enumerate(proposal["confs"]):
  for x,y in enumerate(a):
    if len(y) != 1:
      print("!")
      print(i,x)

print(det_result.keys())
print(proposal.keys())

print(len(det_result["detection_bboxes"][0]))
print(len(det_result["detection_labels"][0]))
print(len(det_result["detection_confs"][0]))

print(len(proposal["boxes"]))
print(len(proposal["cls"]))
print(len(proposal["confs"]))

print()

for i,(lp_boxes, lp_cls, lp_confs, boxes, cls, confs) in \
        enumerate(zip(det_result["detection_bboxes"][0],
                      det_result["detection_labels"][0],
                      det_result["detection_confs"][0],
                      proposal["boxes"],
                      proposal["cls"],
                      proposal["confs"])
      ):
        print("{}:".format(i))
        print((lp_boxes))
        print((lp_boxes))
        print((lp_cls))
        print((lp_confs))
        print((boxes))
        print((cls))
        print((confs))
        if lp_boxes.size == 0:
          if boxes.size != 0:
            print("non-empty + empty!")
            input()
          continue

        if np.max(np.abs((lp_cls-1)-cls)) > 0:
          print("cls:")
          print(lp_cls, cls)
          input()

        if np.max(np.abs((lp_boxes-1)-boxes)) > 0:
          print("boxes:")
          print(lp_boxes, boxes)
          input()

        if np.max(np.abs(lp_confs-confs)) > 0.0001:
          print("confs:")
          print(lp_confs, confs)
          input()
