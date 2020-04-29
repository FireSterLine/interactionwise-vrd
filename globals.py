from os.path import join

images_dir = "images"
models_dir = "models"
data_dir   = "data"

faster_rcnn_dir = "faster-rcnn"
faster_rcnn_models_dir = join(faster_rcnn_dir, "models")


def emb_model_path(model_name):
  if model_name == "gnews":
    w2v_model_path = join(data_dir, "embeddings", "GoogleNews-vectors-negative300.bin.gz")
  elif "coco" in model_name:
    _, ep, dim = model_name.split("-")
    w2v_model_path = "/home/findwise/interactionwise/wikipedia_dump/coco/epoch_{}_dim_{}.model".format(ep, dim)
  elif model_name in ["50", "100", "300"]:
    w2v_model_path = "/home/findwise/interactionwise/wikipedia_dump/epoch_4_dim_{}.model".format(model_name)
  else:
    raise ValueError("Unknown embedding model: '{}'".format(model_name))
  return w2v_model_path

def emb_model_size(model_name):
  if model_name == "gnews":
    emb_size = 300
  elif "coco" in model_name:
    _, ep, dim = model_name.split("-")
    emb_size = int(dim)
  elif model_name in ["50", "100", "300"]:
    emb_size = int(model_name)
  else:
    raise ValueError("Unknown embedding model: '{}'".format(model_name))
  return emb_size
