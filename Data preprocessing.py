import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold


atom_features = pd.read_csv('atom_features_PCA.csv')

smiles = atom_features['SMILES']

bond_features = pd.read_csv('bond_features_PCA.csv')

#seed(s)
smilesF = pd.read_csv('SMILES_200Features.csv')
del smilesF['SMILES']

sheader = []

smilesF = smilesF.fillna(0)

#smilesF2 = smilesF.transpose()
sheader = list(smilesF.columns.values)
scaler = preprocessing.StandardScaler()
scaled = scaler.fit_transform(smilesF)
scaled_df = pd.DataFrame(scaled, columns = sheader)
#scaled_df = scaled_df.transpose()



del atom_features['compound_stem']
del atom_features['SMILES']

atom_features = atom_features.add_prefix('A_')

del bond_features['compound_stem']
del bond_features['SMILES']

bond_features = bond_features.add_prefix('B_')

for i in range (0,12):
  i = str(i)
  column = 'B_'+i
  b1 = bond_features[column]
  
  name = column
  i = int(i)
  atom_features.insert(i,name,b1)

header_smilesf = []
header_smilesf = list(smilesF.columns.values)

#tf.random.set_seed(s)
#seed(s)
second_pca = PCA(n_components = 50) 
data_atom_pca = second_pca.fit_transform(atom_features)

pcaNames = []
for p in range(1,51):
  pc = str(p)
  pca = 'PCA'+pc
  pcaNames.append(pca)


data_atom_pca = pd.DataFrame(data=data_atom_pca, columns=pcaNames)

j = 0
for col in pcaNames:
  col_data = data_atom_pca[col]
  scaled_df.insert(j,col,col_data)
  
  j = j+1


sel = VarianceThreshold(0)
cleaned = sel.fit_transform(features)
