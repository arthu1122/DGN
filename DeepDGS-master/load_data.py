import csv
from itertools import islice

import numpy as np
import pandas as pd
from rdkit import Chem
import networkx as nx



def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                           'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                           'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                           'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    [atom.GetIsAromatic()])


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

#
# def smile_to_graph(smile):
#     mol = Chem.MolFromSmiles(smile)
#
#     c_size = mol.GetNumAtoms()
#
#     features = []
#     for atom in mol.GetAtoms():
#         feature = atom_features(atom)
#         features.append(feature / sum(feature))
#
#     edges = []
#     for bond in mol.GetBonds():
#         edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
#     g = nx.Graph(edges).to_directed()
#     edge_index = []
#     for e1, e2 in g.edges:
#         edge_index.append([e1, e2])
#
#     return c_size, features, edge_index
#
#
# # get drug
# compound_iso_smiles = []
# df = pd.read_csv('data/smiles.csv')
# compound_iso_smiles += list(df['smile'])
# compound_iso_smiles = set(compound_iso_smiles)
# smile_graph = {}
# for smile in compound_iso_smiles:
#     g = smile_to_graph(smile)
#     smile_graph[smile] = g

# get drug
drug_featrues = []
drug_file = 'data/independent_set/Feature_DRUG.csv'
with open(drug_file) as csvfile:
    csv_reader1 = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
    for row in csv_reader1:
        drug_featrues.append(row)
drug_featrues = np.array(drug_featrues)

#get protein
protein_featrues = []
protein_file = 'data/independent_set/Feature_TAR.csv'
with open(protein_file) as csvfile:
    csv_reader2 = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
    for row in csv_reader2:
        protein_featrues.append(row)
protein_featrues = np.array(protein_featrues)

# get cell
cell_featrues = []
cell_file = 'data/independent_set/Feature_CELL.csv'
with open(cell_file) as csvfile:
    csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
    for row in csv_reader:
        cell_featrues.append(row)
cell_featrues = np.array(cell_featrues)



#对三种节点做相同操作。
def get_cell_feature(cellId, cell_features):
    for row in islice(cell_features, 0, None):
        if cellId in row[0]:
            return row[1:]




def load(datafile):
    train_featrues = []
    train_labels = []
    i = 1
    with open(datafile, 'r') as f:
        next(f)

        reader = csv.reader(f)
        for row in reader:
            # c_size, drug1, edge_index = smile_graph[row[0]]
            # drug1 = np.sum(drug1, axis=0)
            #
            # c_size, drug2, edge_index = smile_graph[row[1]]
            # drug2 = np.sum(drug2, axis=0)
            #
            # drug_con = np.hstack((drug1, drug2))
            drug1_featrue=np.array(get_cell_feature(row[0],drug_featrues))
            drug2_featrue=np.array(get_cell_feature(row[1],drug_featrues))
            drug_con = np.hstack((drug1_featrue, drug2_featrue))
            cell_featrue = np.array(get_cell_feature(row[2], cell_featrues))

            feature = np.hstack((drug_con, cell_featrue))
            temp = []
            for i in feature:
                temp.append(float(i))
            train_featrues.append(temp)
            train_labels.append(int(row[3]))

        featrues, labels = np.array(train_featrues), np.array(train_labels)
        print(featrues.shape)
        print(labels.shape)
        return featrues, labels


if __name__ == "__main__":
    a, b = load('data/Label_DRUGCOMB.csv')
    print(a)
    print(b)
    txtDF = pd.DataFrame(data=a)
    txtDF.to_csv('data/Label_DRUGCOMB.csv', index=False, header=False)

    txtDF = pd.DataFrame(data=b)
    txtDF.to_csv('data/Label_DRUGCOMB.csv', index=False, header=False)

