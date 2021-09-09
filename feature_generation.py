import numpy as np
import pandas as pd
import sklearn
import rdkit
from typing import List, Tuple, Union
from rdkit import Chem
from sklearn.decomposition import PCA


def one_hot_encoding(value, choices):
    encoding = [0] * (len(choices))
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1
    return encoding

ATOM_FEATURES = {
    'atomic_num': list(range(118)), # type of atom (ex. C,N,O), by atomic number, size = 118
    'degree': [0, 1, 2, 3, 4, 5], # number of bonds the atom is involved in, size = 6
    'formal_charge': [-1, -2, 1, 2, 0], # integer electronic charge assigned to atom, size = 5
    'chiral_tag': [0, 1, 2, 3], # chirality: unspecified, tetrahedral CW/CCW, or other, size = 4
    'num_Hs': [0, 1, 2, 3, 4], # number of bonded hydrogen atoms, size = 5
    'hybridization': [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2
    ], # size = 5
}

def atom_features_raw(atom):
    features = [atom.GetAtomicNum()] + \
               [atom.GetTotalDegree()] + \
               [atom.GetFormalCharge()] + \
               [int(atom.GetChiralTag())] + \
               [int(atom.GetTotalNumHs())] + \
               [int(atom.GetHybridization())] + \
               [atom.GetIsAromatic()] + \
               [atom.GetMass()]
    return features

def atom_features_onehot(atom): # size: 151
    features = one_hot_encoding(atom.GetAtomicNum() - 1, ATOM_FEATURES['atomic_num']) + \
               one_hot_encoding(atom.GetTotalDegree(), ATOM_FEATURES['degree']) + \
               one_hot_encoding(atom.GetFormalCharge(), ATOM_FEATURES['formal_charge']) + \
               one_hot_encoding(int(atom.GetChiralTag()), ATOM_FEATURES['chiral_tag']) + \
               one_hot_encoding(int(atom.GetTotalNumHs()), ATOM_FEATURES['num_Hs']) + \
               one_hot_encoding(int(atom.GetHybridization()), ATOM_FEATURES['hybridization']) + \
               [1 if atom.GetIsAromatic() else 0] + \
               [atom.GetMass() * 0.01]  # scaled to about the same range as other features
    return features

def bond_features_raw(bond):
    bt = bond.GetBondType()
    if bt == Chem.rdchem.BondType.SINGLE: btt = 0
    elif bt == Chem.rdchem.BondType.DOUBLE: btt = 1
    elif bt == Chem.rdchem.BondType.TRIPLE: btt = 2
    elif bt == Chem.rdchem.BondType.AROMATIC: btt = 3
    fbond = [
        btt,
        (bond.GetIsConjugated() if bt is not None else 0),
        (bond.IsInRing() if bt is not None else 0),
        int(bond.GetStereo())]
    return fbond

def bond_features_onehot(bond: Chem.rdchem.Bond) -> List[Union[bool, int, float]]:
    bt = bond.GetBondType()
    fbond = [
        bt == Chem.rdchem.BondType.SINGLE,
        bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE,
        bt == Chem.rdchem.BondType.AROMATIC,
        (bond.GetIsConjugated() if bt is not None else 0),
        (bond.IsInRing() if bt is not None else 0),
    ]
    fbond += one_hot_encoding(int(bond.GetStereo()), list(range(6)))
    return fbond

class LocalFeatures:
    def __init__(self, mol, onehot = False, pca = False, ids = None):
        if type(mol) == str:
            mol = Chem.MolFromSmiles(mol)

        self.mol = mol
        self.onehot = onehot
        self.n_atoms = 0
        self.n_bonds = 0
        self.f_atoms = []
        self.f_bonds = []
        self.f_atoms_pca = []
        self.f_bonds_pca = []
        self.mol_id_atoms = []
        self.mol_id_bonds = []

        if onehot:
            self.f_atoms = [atom_features_onehot(atom) for atom in mol.GetAtoms()]
            self.f_bonds = [bond_features_onehot(bond) for bond in mol.GetBonds()]
        else:
            self.f_atoms = [atom_features_raw(atom) for atom in mol.GetAtoms()]
            self.f_bonds = [bond_features_raw(bond) for bond in mol.GetBonds()]

        self.n_atoms = len(self.f_atoms)
        self.n_bonds = len(self.f_bonds)
        self.f_atoms_dim = np.shape(self.f_atoms)[1]
        self.f_bonds_dim = np.shape(self.f_bonds)[1]

        if pca:
            fa = np.array(self.f_atoms).T
            fb = np.array(self.f_bonds).T
            pca = PCA(n_components=1)
            pc_atoms = pca.fit_transform(fa)
            pc_bonds = pca.fit_transform(fb)

            self.f_atoms_pca = pc_atoms.T
            self.f_bonds_pca = pc_bonds.T

        if ids is not None:
            self.mol_id_atoms = [ids for i in range(self.n_atoms)]
            self.mol_id_bonds = [ids for i in range(self.n_bonds)]

class BatchLocalFeatures:
    def __init__(self, mol_graphs):
        self.mol_graphs = mol_graphs
        self.n_atoms = 0
        self.n_bonds = 0
        self.a_scope = []
        self.b_scope = []
        f_atoms, f_bonds = [], []
        f_atoms_pca, f_bonds_pca= [], []
        f_atoms_id, f_bonds_id= [], []

        for mol_graph in self.mol_graphs: # for each molecule graph
            f_atoms.extend(mol_graph.f_atoms)
            f_bonds.extend(mol_graph.f_bonds)
            f_atoms_pca.extend(mol_graph.f_atoms_pca)
            f_bonds_pca.extend(mol_graph.f_bonds_pca)

            f_atoms_id.extend(mol_graph.mol_id_atoms)
            f_bonds_id.extend(mol_graph.mol_id_bonds)

            self.a_scope.append((self.n_atoms, mol_graph.n_atoms))
            self.b_scope.append((self.n_bonds, mol_graph.n_bonds))
            self.n_atoms += mol_graph.n_atoms
            self.n_bonds += mol_graph.n_bonds

        self.f_atoms = f_atoms
        self.f_bonds = f_bonds
        self.f_atoms_pca = f_atoms_pca
        self.f_bonds_pca = f_bonds_pca
        self.f_atoms_id = f_atoms_id
        self.f_bonds_id = f_bonds_id

def mol2local(mols, onehot = False, pca = False, ids = None):
    if ids is not None:
        return BatchLocalFeatures([LocalFeatures(mol, onehot, pca, iid) for mol,iid in zip(mols,ids)])
    else:
        return BatchLocalFeatures([LocalFeatures(mol, onehot, pca, ids) for mol in mols])


if __name__ == '__main__':
    data = pd.read_csv("compound-annotation.csv")
    data_smiles = data['SMILES'].values.tolist()
    data_id = data['compound_stem']
    res = mol2local(data_smiles, onehot=True, pca = True, ids = data_id)
    f_atoms_pca = pd.DataFrame(res.f_atoms_pca)
    f_bonds_pca = pd.DataFrame(res.f_bonds_pca)
    f_atoms_pca.to_csv('atom_features_PCA.csv', index=False)
    f_bonds_pca.to_csv('bond_features_PCA.csv', index=False)
    print('Done!')
