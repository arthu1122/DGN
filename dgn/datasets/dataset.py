import time

import torch
from pandas import Series
from torch import tensor
from torch.utils.data import Dataset

from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
import pandas as pd
import re

import numpy as np


# graph data creating
# Heterogeneous graph
def create_data(drug_feature, target_feature, dd_edge_index, dt_edge_index, tt_edge_index, dd_edge_att, device):
    data = HeteroData()

    data['drug'].x = drug_feature  # [num_drugs, num_features_drug]
    data['target'].x = target_feature

    data['drug', 'd-d', 'drug'].edge_index = dd_edge_index  # [2, num_edges_d-d]
    data['drug', 'd-t', 'target'].edge_index = dt_edge_index
    data['target', 't-t', 'target'].edge_index = tt_edge_index

    data['drug', 'd-d', 'drug'].edge_attr = torch.unsqueeze(dd_edge_att, -1)  # [num_edges_d-d, num_features_d-d]
    data['drug', 'd-t', 'target'].edge_attr = torch.ones([dt_edge_index.shape[-1], 1])
    data['target', 't-t', 'target'].edge_attr = torch.ones([tt_edge_index.shape[-1], 1])

    # undirect edge
    data = T.ToUndirected()(data)

    return data


# get encoded id
def get_id(id_dict, ID, is_drug=True):
    if is_drug:
        key = "Drug|||"
    else:
        key = "Target|||"
    return id_dict[key + str(ID)]


def get_graph_data(features_drug_file, features_target_file, e_dd_index_file, e_dt_index_file, e_tt_index_file,
                   e_dd_att_file, device):
    features_drug = pd.read_csv(features_drug_file, index_col='DrugID', dtype=str)
    features_target = pd.read_csv(features_target_file, index_col='TargetID', dtype=str)

    with open(e_dd_index_file, 'r') as f:
        edge_drug_drug_index = f.readlines()
    with open(e_dd_att_file, 'r') as f:
        edge_drug_drug_att = f.readlines()
    with open(e_dt_index_file, 'r') as f:
        edge_drug_target_index = f.readlines()
    with open(e_tt_index_file, 'r') as f:
        edge_target_target_index = f.readlines()

    drug_feature = torch.FloatTensor(features_drug.values.astype('float'))
    target_feature = torch.FloatTensor(features_target.values.astype('float'))

    def transform(indexs):
        # "[0,1]\n" -> [0,1]
        begin = []
        end = []
        for index in indexs:
            begin.append(int(index[1:-2].split(",")[0]))
            end.append(int(index[1:-2].split(",")[1]))
        return torch.tensor([begin, end])

    dd_edge_index = transform(edge_drug_drug_index)
    dt_edge_index = transform(edge_drug_target_index)
    tt_edge_index = transform(edge_target_target_index)

    dd_edge_att = torch.FloatTensor([float(att[:-1]) for att in edge_drug_drug_att])

    graph_data = create_data(drug_feature, target_feature, dd_edge_index, dt_edge_index, tt_edge_index, dd_edge_att,
                             device)

    return graph_data


class GNNDataset(Dataset):
    def __init__(self, label_df, vocab_file, features_cell_df, device):
        # vocab: drug_vocab
        super(GNNDataset, self).__init__()

        # # TODO test
        # label_df = label_df[:100]

        cell_list = list(label_df['Cell_Line_Name'].str.lower())
        drug1_list = list(label_df['DrugID1'])
        drug2_list = list(label_df['DrugID2'])
        label_list = list(label_df['Label'])

        features_cell_df.index = features_cell_df.index.str.lower()

        features_cell_np = features_cell_df.loc[cell_list].to_numpy().astype('float')
        self.features_cell_t = torch.FloatTensor(features_cell_np)\
            # .to(device)
        self.labels_t = torch.LongTensor(label_list)\
            # .to(device)

        with open(vocab_file, 'r') as f:
            self.vocab = f.readlines()
            self.ID2id = {}
            for i in range(len(self.vocab)):
                self.ID2id[self.vocab[i][:-1]] = i

        drug1_id_list = [get_id(self.ID2id, x) for x in drug1_list]
        drug2_id_list = [get_id(self.ID2id, x) for x in drug2_list]

        self.drug1_t = torch.LongTensor(drug1_id_list)\
            # .to(device)
        self.drug2_t = torch.LongTensor(drug2_id_list)\
            # .to(device)

    def __getitem__(self, index):
        return self.drug1_t[index].unsqueeze(-1), self.drug2_t[index].unsqueeze(-1), self.features_cell_t[index], self.labels_t[index].unsqueeze(-1)

    def __len__(self):
        return len(self.labels_t)
