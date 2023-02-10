import csv
from itertools import islice

import pandas as pd
import numpy as np
import os
import json,pickle
from collections import OrderedDict
from rdkit import Chem
# from rdkit.Chem import MolFromSmiles
import networkx as nx
from utils_test import *

#每一行的第一位一般是药物节点，蛋白质节点或者癌细胞节点的名字，后续为其相应特征
def get_cell_feature(cellId, cell_features):
    for row in islice(cell_features, 0, None):
        if row[0] == cellId:
            return row[1: ]
def creat_data(datafile,drugfile,proteinfile,cellfile):
#对三种节点做相同的特征提取操作
    file2 = drugfile
    drug_features = []
    with open(file2) as csvfile:
        csv_reader2 = csv.reader(csvfile)
        for row in csv_reader2:
            drug_features.append(row)
    drug_features = np.array(drug_features)
    print('drug_features', drug_features)
    file3 = proteinfile
    protein_features = []
    with open(file3) as csvfile:
        csv_reader3 = csv.reader(csvfile)
        for row in csv_reader3:
            protein_features.append(row)
    protein_features = np.array(protein_features)
    print('protein_features', protein_features)
    file4=cellfile
    cell_features=[]
    with open(file4) as csvfile:
        csv_reader4=csv.reader(csvfile)
        for row in csv_reader4:
            cell_features.append(row)
    cell_features=np.array(cell_features)
    print('cell_features',cell_features)
    datasets = datafile
    # convert to PyTorch data format
    processed_data_file_train = 'data/processed/' + datasets + '_train.pt'
    #print(processed_data_file_train)
    if ((not os.path.isfile(processed_data_file_train))):
        df = pd.read_csv('data/' + datasets + '.csv')
        drug1, drug2, cell, label = list(df['DrugID1']), list(df['DrugID2']), list(df['Cell_Line_Name']), list(df['Label'])
        print('first',drug1,drug2,cell,label)
        drug1, drug2, cell, label = np.asarray(drug1), np.asarray(drug2), np.asarray(cell), np.asarray(label)
        print('second', drug1, drug2, cell, label)
        # make data PyTorch Geometric ready
        print('开始创建数据')
        print('创建数据成功')
        print('preparing ', datasets + '_.pt in pytorch format!')

if __name__ == "__main__":
    # datafile = 'prostate'
    drugfile='data\independent_set\Feature_DRUG.csv'
    proteinfile= 'data\independent_set\Feature_TAR.csv'
    cellfile = 'data\independent_set\Feature_CELL.csv'
    da = ['Label_DRUGCOMB']
    for datafile in da:
        creat_data(datafile, drugfile,proteinfile,cellfile)