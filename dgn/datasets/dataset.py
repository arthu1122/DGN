import torch
from torch import tensor
from torch.utils.data import Dataset

from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
import pandas as pd
import re


# graph data creating
def create_data(drug_feature, target_feature, dd_edge_index, dt_edge_index, tt_edge_index, dd_edge_att):
    data = HeteroData()

    data['drug'].x = drug_feature  # [num_drugs, num_features_drug]
    data['target'].x = target_feature

    data['drug', 'd-d', 'drug'].edge_index = dd_edge_index  # [2, num_edges_d-d]
    data['drug', 'd-t', 'target'].edge_index = dt_edge_index
    # data['target', 't-d', 'drug'].edge_index = dt_edge_index[::-1]
    data['target', 't-t', 'target'].edge_index = tt_edge_index

    data['drug', 'd-d', 'drug'].edge_attr = torch.unsqueeze(dd_edge_att,-1)   # [num_edges_d-d, num_features_d-d]
    data['drug', 'd-t', 'target'].edge_attr = torch.ones([dt_edge_index.shape[-1], 1])
    # data['target', 't-d', 'drug'].edge_attr = torch.ones([dt_edge_index.shape[-1],1])
    data['target', 't-t', 'target'].edge_attr = torch.ones([tt_edge_index.shape[-1], 1])

    data = T.ToUndirected()(data)

    return data


class GNNDataset(Dataset):
    def __init__(self, label_file, features_drug_file, features_target_file, features_cell_file, vocab_file,
                 e_dd_index_file, e_dd_att_file, e_dt_index_file, e_tt_index_file):
        super(GNNDataset, self).__init__()

        df = pd.read_csv(label_file)
        self.data = df[['Cell_Line_Name', 'DrugID1', 'DrugID2']]
        self.label = list(df['Label'])

        self.features_drug = pd.read_csv(features_drug_file, index_col='DrugID', dtype=str)
        self.features_target = pd.read_csv(features_target_file, index_col='TargetID', dtype=str)
        self.features_cell = pd.read_csv(features_cell_file, index_col='Cell_Line_Name', dtype=str)

        with open(vocab_file, 'r') as f:
            self.vocab = f.readlines()

        with open(e_dd_index_file, 'r') as f:
            self.edge_drug_drug_index = f.readlines()

        with open(e_dd_att_file, 'r') as f:
            self.edge_drug_drug_att = f.readlines()

        with open(e_dt_index_file, 'r') as f:
            self.edge_drug_target_index = f.readlines()

        # with open(e_dt_att_file, 'r') as f:
        #     self.edge_drug_target_att = f.readlines()

        with open(e_tt_index_file, 'r') as f:
            self.edge_target_target_index = f.readlines()

        # with open(e_tt_att_file, 'r') as f:
        #     self.edge_target_target_att = f.readlines()

    def __getitem__(self, index):
        cell_name, drug1, drug2 = self.data.iloc[index]
        drug1_feature = list(self.features_drug.astype('float').loc[drug1])
        drug2_feature = list(self.features_drug.astype('float').loc[drug2])
        cell_feature = list(self.features_cell.astype('float').loc[cell_name])

        drug1_feature = torch.FloatTensor(drug1_feature)
        drug2_feature = torch.FloatTensor(drug2_feature)
        cell_feature = torch.FloatTensor(cell_feature)

        return (drug1_feature, drug2_feature, cell_feature), self.label[index]

    def get_graph_data(self):
        drug_feature = torch.FloatTensor(self.features_drug.values.astype('float'))
        target_feature = torch.FloatTensor(self.features_target.values.astype('float'))

        # "[0,1]\n" -> [0,1]
        begin=[]
        end=[]
        for index in self.edge_drug_drug_index:
            begin.append(int(index[1:-2].split(",")[0]))
            end.append(int(index[1:-2].split(",")[1]))
        dd_edge_index = torch.tensor([begin,end])

        begin=[]
        end=[]
        for index in self.edge_drug_target_index:
            begin.append(int(index[1:-2].split(",")[0]))
            end.append(int(index[1:-2].split(",")[1]))
        dt_edge_index = torch.tensor([begin,end])

        begin=[]
        end=[]
        for index in self.edge_target_target_index:
            begin.append(int(index[1:-2].split(",")[0]))
            end.append(int(index[1:-2].split(",")[1]))
        tt_edge_index = torch.tensor([begin,end])


        dd_edge_att = torch.FloatTensor([float(att[:-1]) for att in self.edge_drug_drug_att])

        graph_data = create_data(drug_feature, target_feature, dd_edge_index, dt_edge_index, tt_edge_index, dd_edge_att)

        return graph_data


if __name__ == '__main__':
    dataset = GNNDataset(label_file="../../data/raw/Label_DRUGCOMB.csv",
                         features_drug_file="../../data/raw/Feature_DRUG.csv",
                         features_target_file="../../data/raw/Feature_TAR.csv",
                         features_cell_file="../../data/raw/Feature_CELL.csv",
                         vocab_file="../../data/processed/vocab.txt",
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
