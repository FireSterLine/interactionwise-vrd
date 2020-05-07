import json
import numpy as np

with open('tuples-counts_train.json', 'r') as f:
  counts = json.load(f)


with open('tuples-counts_test.json', 'r') as f:
  counts_test = json.load(f)


with open('tuples-counts_test_zs.json', 'r') as f:
	counts_test_zs = json.load(f)


a = {}

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

len(counts)
len(counts_test)
len(counts_test_zs)
len(a)

np.savetxt("tuple_counts.csv", np.array(tuple_counts), delimiter=",", fmt="%s")
