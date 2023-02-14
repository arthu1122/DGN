import torch
from pandas import Series
from torch import tensor
from torch.utils.data import Dataset

from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
import pandas as pd
import re


# graph data creating
def create_data(drug_feature, target_feature, dd_edge_index, dt_edge_index, tt_edge_index, dd_edge_att, device):
    drug_feature = drug_feature.to(device)
    target_feature = target_feature.to(device)
    dd_edge_index = dd_edge_index.to(device)
    dt_edge_index = dt_edge_index.to(device)
    tt_edge_index = tt_edge_index.to(device)
    dd_edge_att = dd_edge_att.to(device)

    data = HeteroData()

    data['drug'].x = drug_feature  # [num_drugs, num_features_drug]
    data['target'].x = target_feature

    data['drug', 'd-d', 'drug'].edge_index = dd_edge_index  # [2, num_edges_d-d]
    data['drug', 'd-t', 'target'].edge_index = dt_edge_index
    # data['target', 't-d', 'drug'].edge_index = dt_edge_index[::-1]
    data['target', 't-t', 'target'].edge_index = tt_edge_index

    data['drug', 'd-d', 'drug'].edge_attr = torch.unsqueeze(dd_edge_att, -1)  # [num_edges_d-d, num_features_d-d]
    data['drug', 'd-t', 'target'].edge_attr = torch.ones([dt_edge_index.shape[-1], 1], device=device)
    # data['target', 't-d', 'drug'].edge_attr = torch.ones([dt_edge_index.shape[-1],1])
    data['target', 't-t', 'target'].edge_attr = torch.ones([tt_edge_index.shape[-1], 1], device=device)

    data = T.ToUndirected()(data)

    return data


def get_id(id_dict, ID, is_drug=True):
    if is_drug:
        key = "Drug|||"
    else:
        key = "Target|||"

    return id_dict[key + str(ID)]


def get_graph_data(features_drug_file, features_target_file, e_dd_index_file, e_dt_index_file, e_tt_index_file,
                   e_dd_att_file,device):
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

    graph_data = create_data(drug_feature, target_feature, dd_edge_index, dt_edge_index, tt_edge_index, dd_edge_att,device)

    return graph_data


class GNNDataset(Dataset):
    def __init__(self, label_df, vocab_file, features_cell_df):
        super(GNNDataset, self).__init__()

        # # TODO test
        # label_df = label_df[:100]

        self.data = label_df[['Cell_Line_Name', 'DrugID1', 'DrugID2']]
        self.label = list(label_df['Label'])

        # self.features_cell = pd.read_csv(features_cell_file, index_col='Cell_Line_Name', dtype=str)
        self.features_cell = features_cell_df
        self.features_cell.index = self.features_cell.index.str.lower()

        with open(vocab_file, 'r') as f:
            self.vocab = f.readlines()
            self.id_dict = {}
            for i in range(len(self.vocab)):
                self.id_dict[self.vocab[i][:-1]] = i

    def __getitem__(self, index):
        index_data = self.data.iloc[index]
        cell_name, drug1, drug2 = index_data['Cell_Line_Name'], index_data['DrugID1'], index_data['DrugID2']
        drug1_id = get_id(self.id_dict, drug1)
        drug2_id = get_id(self.id_dict, drug2)

        cell_name = cell_name.lower()

        # try:
        #     cell_feature = list(self.features_cell.loc[cell_name])
        # except KeyError as e:

        cell_feature = list(self.features_cell.astype('float').loc[cell_name])
        cell_feature = torch.FloatTensor(cell_feature)

        return drug1_id, drug2_id, cell_feature, self.label[index]

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    dataset = GNNDataset(label_file="../../data/processed/Shuffled_Label_Data.csv",
                         features_drug_file="../../data/raw/Feature_DRUG.csv",
                         features_target_file="../../data/raw/Feature_TAR.csv",
                         features_cell_file="../../data/raw/Feature_CELL.csv",
                         vocab_file="../../data/processed/drug_vocab.txt",
                         e_dd_index_file="../../data/processed/drug_drug_edge_index.txt",
                         e_dd_att_file="../../data/processed/drug_drug_edge_att.txt",
                         e_dt_index_file="../../data/processed/drug_tar_edge_index.txt",
                         e_tt_index_file="../../data/processed/tar_tar_edge_index.txt"
                         )
    data = dataset.get_graph_data()

    for d in dataset:
        print(d)
        print(1)

    print(0)
