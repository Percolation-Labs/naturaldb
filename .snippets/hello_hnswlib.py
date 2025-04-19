import hnswlib
import numpy as np

dim = 128
num_elements = 10000

data = np.random.rand(num_elements, dim).astype(np.float32)

p = hnswlib.Index(space='l2', dim=dim)
p.init_index(max_elements=num_elements, ef_construction=200, M=16)
p.add_items(data)
labels, distances = p.knn_query(data[:1], k=5)

print("Nearest neighbors:", labels)