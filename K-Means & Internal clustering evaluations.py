from sklearn import metrics
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans




kmeans = KMeans(50,random_state=21)
kmeans.fit(loaded_array)
labels = kmeans.labels_



ch = metrics.calinski_harabasz_score(cleaned, labels)
print('ch Score: %.3f' % ch)

ss = silhouette_score(cleaned, labels, metric='euclidean')
print('Silhouetter Score: %.3f' % ss)

DB = davies_bouldin_score(cleaned, labels)
print('DB Score: %.3f' % DB)
