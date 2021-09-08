from sklearn.cluster import Birch
brc = Birch(threshold=0.5, branching_factor=50, n_clusters=30, compute_labels=True, copy=True)
brc.fit(cleaned)
labels = brc.labels_
