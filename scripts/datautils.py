import json
import numpy as np
from collections import defaultdict

def add_heatmap_labels(matrix, data_dir, indices = None):
	filename = "{}/predicates.json".format(data_dir)
	with open(filename, 'r') as f:
		predicates = json.load(f)
	#
	if indices is not None:
		predicates = np.array(predicates)[indices].tolist()
	b = [[predicates[i], predicates[i]] + row for i,row in enumerate(matrix.tolist())]
	return np.array(b)

def save_pred2pred(data_dir, model):
  filename = "{}/predicates-emb-{}.json".format(data_dir, model)
  with open(filename, 'r') as f:
    pred_emb = json.load(f)
  pred_emb = np.array(pred_emb)
  from sklearn.metrics.pairwise import cosine_similarity
  pred2pred_sim = cosine_similarity(pred_emb, pred_emb)
  np.savetxt("{}/pred2pred_sim-{}.csv".format(data_dir, model), add_heatmap_labels(pred2pred_sim, data_dir), delimiter=",", fmt="%s")
  indices = np.argsort(pred2pred_sim.sum(axis=1))[::-1]
  pred2pred_sim_sorted = pred2pred_sim[indices][:,indices]
  np.savetxt("{}/pred2pred_sim-sorted-{}.csv".format(data_dir, model), add_heatmap_labels(pred2pred_sim_sorted, data_dir, indices=indices), delimiter=",", fmt="%s")
  return pred2pred_sim


def save_pred2pred_diff(data_dir, model1, model2):
  pred2pred1_sim = save_pred2pred(data_dir, model1)
  pred2pred2_sim = save_pred2pred(data_dir, model2)
  np.savetxt("{}/pred2pred_sim-{}-{}.csv".format(data_dir, model1, model2), add_heatmap_labels((pred2pred1_sim-pred2pred2_sim), data_dir), delimiter=",", fmt="%s")


def save_tuple_counts(data_dir):
	filename = "{}/predicates.json".format(data_dir)
	with open(filename, 'r') as f:
		predicates = json.load(f)

	a = {}
	with open("{}/tuples-counts_train.json".format(data_dir), 'r') as f:
	  counts_train = json.load(f)
	with open("{}/tuples-counts_test.json".format(data_dir), 'r') as f:
	  counts_test = json.load(f)
	with open("{}/tuples-counts_test_zs.json".format(data_dir), 'r') as f:
	  counts_test_zs = json.load(f)
	for tuple_str,count in counts_train.items():
		a[tuple_str] = [count, 0, 0]
	for tuple_str,count in counts_test.items():
		if a.get(tuple_str) is None:
			a[tuple_str] = [0, count, 0]
		else:
			a[tuple_str][1] = count
	for tuple_str,count in counts_test_zs.items():
		if a.get(tuple_str) is None:
			a[tuple_str] = [0, 0, count]
		else:
			a[tuple_str][2] = count

	soP_counts         = defaultdict(lambda: defaultdict(lambda: int(0)))
	soP_counts_train   = defaultdict(lambda: defaultdict(lambda: int(0)))
	soP_counts_test   = defaultdict(lambda: defaultdict(lambda: int(0)))
	sP_counts         = defaultdict(lambda: defaultdict(lambda: int(0)))
	sP_counts_train   = defaultdict(lambda: defaultdict(lambda: int(0)))
	sP_counts_test   = defaultdict(lambda: defaultdict(lambda: int(0)))
	tuple_counts = []
	for tuple_str,counts in a.items():
		tuple_name = tuple_str.replace(",", "")
		train_count   = counts[0]
		test_count    = counts[1]
		test_zs_count = counts[2]
		tot_count = train_count+test_count

		sop_tuple = list(eval(tuple_str))
		soP_counts      [sop_tuple[2]][str((sop_tuple[0], sop_tuple[1]))] += tot_count
		soP_counts_train[sop_tuple[2]][str((sop_tuple[0], sop_tuple[1]))] += train_count
		soP_counts_test [sop_tuple[2]][str((sop_tuple[0], sop_tuple[1]))] += test_count
		sP_counts       [sop_tuple[2]][str(sop_tuple[0])] += tot_count
		sP_counts_train [sop_tuple[2]][str(sop_tuple[0])] += train_count
		sP_counts_test  [sop_tuple[2]][str(sop_tuple[0])] += test_count

		row = [tuple_name, train_count, test_count, tot_count, test_zs_count]
		tuple_counts.append(row)
	print("counts: ", len(counts_train))
	print("counts_test: ", len(counts_test))
	print("counts_test_zs: ", len(counts_test_zs))
	print("a: ", len(a))
	np.savetxt("{}/tuple_counts.csv".format(data_dir), np.array(tuple_counts), delimiter=",", fmt="%s")

	# soP_cooccurrence
	def get_soP_cooccurrence(soP_counts):
		soP_cooccurrence = np.zeros((len(predicates), len(predicates)), dtype=np.int)
		for pred_i in soP_counts:
			for pred_j in soP_counts:
				n_cooccur = 0
				for so_str in soP_counts[pred_i]:
					if so_str in soP_counts[pred_j]:
						n_cooccur += min(soP_counts[pred_j][so_str], soP_counts[pred_i][so_str])
				soP_cooccurrence[pred_i][pred_j] = n_cooccur
		return soP_cooccurrence

	np.savetxt("{}/soP_cooccurrence.csv".format(data_dir), add_heatmap_labels(get_soP_cooccurrence(soP_counts), data_dir), delimiter=",", fmt="%s")
	np.savetxt("{}/soP_cooccurrence-train.csv".format(data_dir), add_heatmap_labels(get_soP_cooccurrence(soP_counts_train), data_dir), delimiter=",", fmt="%s")
	np.savetxt("{}/soP_cooccurrence-test.csv".format(data_dir), add_heatmap_labels(get_soP_cooccurrence(soP_counts_test), data_dir), delimiter=",", fmt="%s")

	# sP_cooccurrence
	def get_sP_cooccurrence(sP_counts):
		sP_cooccurrence = np.zeros((len(predicates), len(predicates)), dtype=np.int)
		for pred_i in sP_counts:
			for pred_j in sP_counts:
				n_cooccur = 0
				for sub_str in sP_counts[pred_i]:
					if sub_str in sP_counts[pred_j]:
						n_cooccur += min(sP_counts[pred_j][sub_str], sP_counts[pred_i][sub_str])
				sP_cooccurrence[pred_i][pred_j] = n_cooccur
		return sP_cooccurrence

	np.savetxt("{}/sP_cooccurrence.csv".format(data_dir), add_heatmap_labels(get_soP_cooccurrence(soP_counts), data_dir), delimiter=",", fmt="%s")
	np.savetxt("{}/sP_cooccurrence-train.csv".format(data_dir), add_heatmap_labels(get_soP_cooccurrence(soP_counts_train), data_dir), delimiter=",", fmt="%s")
	np.savetxt("{}/sP_cooccurrence-test.csv".format(data_dir), add_heatmap_labels(get_soP_cooccurrence(soP_counts_test), data_dir), delimiter=",", fmt="%s")

	return a

"""
save_pred2pred("data/vrd/all", "gnews");
save_pred2pred("data/vrd/all", "300");
save_pred2pred_diff("data/vrd/all", "gnews", "300");
save_pred2pred_diff("data/vrd/all", "50", "300");
save_pred2pred_diff("data/vrd/all", "50", "100");

save_pred2pred("data/vrd/all", "glove-50")
save_pred2pred_diff("data/vrd/all", "glove-50", "50")

save_tuple_counts("data/vrd/all")

"""
