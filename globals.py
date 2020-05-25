from os.path import join

images_dir = "images"
models_dir = "models"
data_dir   = "data"

faster_rcnn_dir = "faster-rcnn"
faster_rcnn_models_dir = join(faster_rcnn_dir, "models")


def emb_model_path(model_name):
  if model_name == "gnews":
    w2v_model_path = join(data_dir, "embeddings", "GoogleNews-vectors-negative300.bin.gz")
  elif "gloco" in model_name:
    _, ep, dim = model_name.split("-")
    w2v_model_path = "/home/findwise/interactionwise/wikipedia_dump/coco/glove_epoch_{}_dim_{}.json".format(int(ep)+5, dim)
  elif "glove" in model_name:
    _, dim = model_name.split("-")
    if int(dim) == 300:
      w2v_model_path = "/home/findwise/interactionwise/wikipedia_dump/glove_epoch_5_dim_300_embeddings.json"
    else:
      w2v_model_path = "/home/findwise/interactionwise/wikipedia_dump/glove_epoch_5_dim_{}.model".format(dim)
  elif "coco" in model_name:
    _, ep, dim = model_name.split("-")
    w2v_model_path = "/home/findwise/interactionwise/wikipedia_dump/coco/word2vec_epoch_{}_dim_{}.model".format(int(ep)+5, dim)
  elif model_name in ["50", "100", "300"]:
    w2v_model_path = "/home/findwise/interactionwise/wikipedia_dump/word2vec_epoch_5_dim_{}.model".format(model_name)
  else:
    raise ValueError("Unknown embedding model: '{}'".format(model_name))
  return w2v_model_path

def emb_model_size(model_name):
  if model_name == "gnews":
    emb_size = 300
  elif "gloco" in model_name:
    _, ep, dim = model_name.split("-")
    emb_size = int(dim)
  elif "glove" in model_name:
    _, dim = model_name.split("-")
    emb_size = int(dim)
  elif "coco" in model_name:
    _, ep, dim = model_name.split("-")
    emb_size = int(dim)
  elif model_name in ["50", "100", "300"]:
    emb_size = int(model_name)
  else:
    raise ValueError("Unknown embedding model: '{}'".format(model_name))
  return emb_size
