from os.path import join

images_dir = "images"
models_dir = "models"
data_dir   = "data"

faster_rcnn_dir = "faster-rcnn"
faster_rcnn_models_dir = join(faster_rcnn_dir, "models")

#embedding_model = "gnews"
#embedding_model = "50"
#embedding_model = "100"
embedding_model = "coco-50"

if embedding_model == "gnews":
  w2v_model_path = join(data_dir, "embeddings", "GoogleNews-vectors-negative300.bin.gz")
  emb_size = 300
elif "coco" in embedding_model:
  w2v_model_path = "/home/findwise/interactionwise/wikipedia_dump/coco/epoch_34_dim_{}.model".format(embedding_model[5:])
  emb_size = int(embedding_model[5:])
elif embedding_model in ["50", "100"]:
  w2v_model_path = "/home/findwise/interactionwise/wikipedia_dump/epoch_4_dim_{}.model".format(embedding_model)
  emb_size = int(embedding_model)
