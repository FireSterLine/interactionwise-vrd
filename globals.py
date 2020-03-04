from os.path import join

images_dir = "images"
images_det_dir = "images_det"
models_dir = "models"
data_dir = "data"
vrd_dir = "vrd"
vg_dir = "vg"

vrd_images_train_dir = "sg_dataset/sg_train_images/"
vrd_images_test_dir = "sg_dataset/sg_test_images/"

vrd_train_file = "vrd_data.json"
vrd_objects_vocab_file = "objects.json"
vrd_predicates_vocab_file = "predicates.json"

vg_objects_vocab_file = "objects_vocab.txt"
vg_predicates_vocab_file = "relations_vocab.txt"

faster_rcnn_dir = "faster-rcnn"
faster_rcnn_models_dir = join(faster_rcnn_dir, "models")

w2v_model_path = "word2vec_model/GoogleNews-vectors-negative300.bin.gz"
