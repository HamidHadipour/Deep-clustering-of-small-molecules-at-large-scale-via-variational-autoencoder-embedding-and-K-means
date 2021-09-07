from numpy.random import seed 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

#seed(s)
km_scores= []
vae32 = []
db_score = []
for i in range(5,205,5):
    #seed(130)
    km = KMeans(n_clusters=i, random_state=25).fit(pred)
    #seed(130)
    km_preds = km.predict(pred)
    

    #seed(130)
    silhouette = silhouette_score(pred,km_preds,random_state=25)
    vae32_hv3_1000e.append(silhouette)
    print("Silhouette score for number of cluster(s) {}: {}".format(i,silhouette))


plt.figure(figsize=(250,80))
plt.title("",fontsize=96)
plt.scatter(x=[i for i in range(5,205,5)],y=vae32_hv3_1000e,s=6000,edgecolor='k', color = 'blue')
plt.grid(True, linewidth=15)
plt.xlabel("\n\nNumber of clusters",fontsize=180)
plt.ylabel("Silhouette score\n",fontsize=180)
plt.xticks([i for i in range(5,205,5)],fontsize=150)
plt.yticks(fontsize=150)
plt.show()
