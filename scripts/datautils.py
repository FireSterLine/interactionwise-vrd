import json
import numpy as np

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
	a = {}

	with open("{}/tuples-counts_train.json".format(data_dir), 'r') as f:
	  counts = json.load(f)
	with open("{}/tuples-counts_test.json".format(data_dir), 'r') as f:
	  counts_test = json.load(f)
	with open("{}/tuples-counts_test_zs.json".format(data_dir), 'r') as f:
		counts_test_zs = json.load(f)

	for tuple_str,count in counts.items():
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

	tuple_counts = []
	for tuple_str,counts in a.items():
		row = [tuple_str.replace(",", "")]
		row += [counts[0]] + [counts[1]] + [counts[0]+counts[1]] + [counts[2]]
		tuple_counts.append(row)
	len(counts, counts_test, counts_test_zs, a)
	np.savetxt("{}/tuple_counts.csv".format(data_dir), np.array(tuple_counts), delimiter=",", fmt="%s")

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
