import json
import numpy as np
from collections import defaultdict

def add_heatmap_labels(matrix, data_dir, indices = None):
	filename = "{}/predicates.json".format(data_dir)
	with open(filename, 'r') as f:
		predicates = json.load(f)

	# if "activities" in data_dir and len(predicates) == 28:
	# 	idx = np.array(list(range(0, 15)) + list(range(16, len(predicates))))
	# 	print(matrix[idx].shape)
	# 	print(matrix[idx][:,idx].shape)
	# 	matrix = matrix[idx][:,idx]
	# 	print(matrix.shape)
	# 	del predicates[15]

	# Avoid bright spots
	# for i in range(len(predicates)):
	# 	matrix[i][i] = 0

	if indices is not None:
		predicates = np.array(predicates)[indices].tolist()
	b = [[predicates[i], predicates[i]] + row for i,row in enumerate(matrix.tolist())]
	return np.array(b)

def save_pred2pred(data_dir, model, force_indices = None, ind_method = "sum"):
	filename = "{}/predicates.json".format(data_dir)
	with open(filename, 'r') as f:
		predicates = json.load(f)
	filename = "{}/predicates-emb-{}.json".format(data_dir, model)
	with open(filename, 'r') as f:
	    pred_emb = json.load(f)
	pred_emb = np.array(pred_emb)
	from sklearn.metrics.pairwise import cosine_similarity
	pred2pred_sim = cosine_similarity(pred_emb, pred_emb)
	np.savetxt("{}/pred2pred_sim-{}.csv".format(data_dir, model), add_heatmap_labels(pred2pred_sim, data_dir), delimiter=",", fmt="%s")
	if ind_method   == "sum": indices = np.argsort(pred2pred_sim.sum(axis=1))[::-1]
	elif ind_method == "std": indices = np.argsort(pred2pred_sim.std(axis=1))[::-1]
	elif ind_method == "mwe": indices = np.argsort(np.array([x.count(" ") for x in predicates]))[::-1]
	# indices = np.argsort(np.array([x.count(" ")*10000+sum for x,sum in zip(predicates,pred2pred_sim.sum(axis=1))]))[::-1]
	pred2pred_sim_sorted = pred2pred_sim[indices][:,indices]
	np.savetxt("{}/pred2pred_sim-sorted-{}.csv".format(data_dir, model), add_heatmap_labels(pred2pred_sim_sorted, data_dir, indices=indices), delimiter=",", fmt="%s")
	if force_indices is not None:
		indices = force_indices
		pred2pred_sim_sorted = pred2pred_sim[indices][:,indices]
	return pred2pred_sim, pred2pred_sim_sorted, indices


def save_pred2pred_diff(data_dir, model1, model2):
  sort_methods = ["mwe", "sum"] # Sort lines by multi-word-expr-or-not and sum
  for sort_method in sort_methods:
	  pred2pred1_sim, pred2pred1_sim_sorted, indices = save_pred2pred(data_dir, model1, ind_method = sort_method)
	  pred2pred2_sim, pred2pred2_sim_sorted, _       = save_pred2pred(data_dir, model2, indices)

	  np.savetxt("{}/pred2pred_sim-{}-{}-diff.csv".format(data_dir, model1, model2),
	    add_heatmap_labels(np.abs(pred2pred1_sim-pred2pred2_sim), data_dir), delimiter=",", fmt="%s")

	  np.savetxt("{}/pred2pred_sim-sorted-{}-{}(sorted-{}).csv".format(data_dir, sort_method, model1, model2),
	    add_heatmap_labels(pred2pred1_sim_sorted, data_dir, indices = indices), delimiter=",", fmt="%s")
	  np.savetxt("{}/pred2pred_sim-sorted-{}-{}(sorted-{}).csv".format(data_dir, sort_method, model2, model1),
	    add_heatmap_labels(pred2pred2_sim_sorted, data_dir, indices = indices), delimiter=",", fmt="%s")
	  np.savetxt("{}/pred2pred_sim-sorted-{}-{}-{}-diff.csv".format(data_dir, sort_method, model1, model2),
	    add_heatmap_labels(np.abs(pred2pred1_sim_sorted-pred2pred2_sim_sorted), data_dir, indices = indices), delimiter=",", fmt="%s")


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

	np.savetxt("{}/sP_cooccurrence.csv".format(data_dir), add_heatmap_labels(get_sP_cooccurrence(sP_counts), data_dir), delimiter=",", fmt="%s")
	np.savetxt("{}/sP_cooccurrence-train.csv".format(data_dir), add_heatmap_labels(get_sP_cooccurrence(sP_counts_train), data_dir), delimiter=",", fmt="%s")
	np.savetxt("{}/sP_cooccurrence-test.csv".format(data_dir), add_heatmap_labels(get_sP_cooccurrence(sP_counts_test), data_dir), delimiter=",", fmt="%s")

	return a

save_pred2pred("data/vrd/all",        "gnews");
save_pred2pred("data/vrd/spatial",    "gnews");
save_pred2pred("data/vrd/activities", "gnews");
save_pred2pred("data/genome/150-50-50/all",        "gnews");
save_pred2pred("data/genome/150-50-50/spatial",    "gnews");
save_pred2pred("data/genome/150-50-50/activities", "gnews");


save_pred2pred("data/vrd/activities",        "glove-300");
save_pred2pred("data/vrd/spatial",        "glove-300");
save_pred2pred("data/vrd/all", "glove-300");
save_pred2pred("data/vrd/activities", "300");
save_pred2pred("data/vrd/spatial",    "300");
save_pred2pred("data/vrd/all", "300");

save_pred2pred_diff("data/vrd/all",                     "gnews", "300");
save_pred2pred_diff("data/vrd/all",                     "gnews", "glove-300");
save_pred2pred_diff("data/vrd/spatial",                 "gnews", "300");
save_pred2pred_diff("data/vrd/spatial",                 "gnews", "glove-300");
save_pred2pred_diff("data/vrd/activities",              "gnews", "300");
save_pred2pred_diff("data/vrd/activities",              "gnews", "glove-300");
save_pred2pred_diff("data/genome/150-50-50/all",        "gnews", "300");
save_pred2pred_diff("data/genome/150-50-50/all",        "gnews", "glove-300");
save_pred2pred_diff("data/genome/150-50-50/spatial",    "gnews", "300");
save_pred2pred_diff("data/genome/150-50-50/spatial",    "gnews", "glove-300");
save_pred2pred_diff("data/genome/150-50-50/activities", "gnews", "300");
save_pred2pred_diff("data/genome/150-50-50/activities", "gnews", "glove-300");

"""
save_pred2pred("data/vrd/all", "gnews");
save_pred2pred("data/vrd/all", "300");
save_pred2pred_diff("data/vrd/all", "gnews", "300");
save_pred2pred_diff("data/vrd/all", "50", "300");
save_pred2pred_diff("data/vrd/all", "50", "100");

save_pred2pred("data/vrd/all", "glove-50")
save_pred2pred_diff("data/vrd/all", "glove-50", "50")

save_tuple_counts("data/vrd/all")
save_tuple_counts("data/vrd/spatial")
save_tuple_counts("data/vrd/activities")
save_tuple_counts("data/genome/150-50-50/all")
save_tuple_counts("data/genome/150-50-50/spatial")
save_tuple_counts("data/genome/150-50-50/activities")

"""
