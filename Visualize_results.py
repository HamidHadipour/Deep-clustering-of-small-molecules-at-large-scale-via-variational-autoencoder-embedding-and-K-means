from keras.models import load_model
import numpy as np
import pandas as pd
from numpy.random import seed
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
# 1. Density Plot
pred = pd.DataFrame(data = loaded_array, columns=predcolumns)

pred.insert(0,'SMILES',smiles)

pred.insert(1,"clusters",labels)


s=[]
i=0
index = []

for j in range(0,50):
  s = pred.loc[pred['clusters']==j,'SMILES']
  s = s.to_list()
  for s2 in s:
    i = i+1
  index.append(i)
  print(i)
  i = 0
  
###################################

plt.rcParams.update({'font.size': 22})

x = []
for i in range(1,51):
    x.append(i)

font = {'size'   : 22}
fig = plt.figure(figsize=(15,10))
ax = fig.add_axes([0,0,1,1])
langs = x
students = index
ax.bar(langs,students)
ax.set_ylabel('Number of molecules in each cluster\n', fontsize = 34)
ax.set_xlabel('\nNumber of clusters ', fontsize = 34)
ax.set_xticks([k for k in range(1,51,5)])
#ax.set_yticks( fontsize = 20)
ax.set_title('', fontsize = 14)

plt.show()

#######################################################################

# 2. t-SNE plot
tsne_comp = TSNE(n_components=2, perplexity=30,random_state=30, n_iter=1000).fit_transform(loaded_array)

tsne_df = pd.DataFrame(data = tsne_comp, columns =['t-SNE1','t-SNE2'])
tsne_df.head()

tsne_df = pd.concat([tsne_df,pd.DataFrame({'cluster':labels})], axis = 1)
tsne_df['cluster']+=1
tsne_df.head()

text = []
for i in range (1,51):
  text.append(str(i))
len(text)

plt.figure(figsize=(25,20))
sns.set(font_scale=3)
z = sns.color_palette("coolwarm", as_cmap=True)
ax = sns.scatterplot(x="t-SNE1", y="t-SNE2", hue = "cluster", data = tsne_df, palette=z)
#ax = sns.color_palette("mako", as_cmap=True)
x = tsne_df['t-SNE1']
y = tsne_df['t-SNE2']
for i in range(0, 50):

  plt.annotate(text[i], (x[i], y[i]+.2 ) ,size=22, color='black', weight='bold')
#plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, ncol = 3 )

plt.show()
