import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GATConv


class HeteroGNNs(nn.Module):
    def __init__(self, num_layers=1, hidden_channels=768):
        super().__init__()

        # graph network
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                ('drug', 'd-d', 'drug'): GATConv((-1, -1), hidden_channels),
                ('drug', 'd-t', 'target'): GATConv((-1, -1), hidden_channels, add_self_loops=False),
                ('target', 'rev_d-t', 'drug'): GATConv((-1, -1), hidden_channels, add_self_loops=False),
                ('target', 't-t', 'target'): GATConv((-1, -1), hidden_channels, ),
            }, aggr='sum')
            self.convs.append(conv)

    def forward(self, x_dict,edge_index_dict):
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: x.relu() for key, x in x_dict.items()}

        return x_dict


class Reduction(nn.Module):
    def __init__(self, input_size, output_size, hidden_size1=2048, hidden_size2=512, dropout=0.2):
        super().__init__()
        self.reduction = nn.Sequential(
            nn.Linear(input_size, hidden_size1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size1, hidden_size2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size2, output_size),
            nn.ReLU()
        )

    def forward(self, x):
        return self.reduction(F.normalize(x, 2, 1))


class UnnamedModel(nn.Module):
    def __init__(self, target_features_num=570, drug_features_num=200, num_features_cell=890, hidden_channels=768,
                 dropout=0.2, mae=True):
        super().__init__()

        # gnn
        self.gnn = HeteroGNNs()

        # get cell features
        self.reduction = Reduction(num_features_cell, hidden_channels * 2)

        # final transform
        self.reduction2 = Reduction(hidden_channels * 4, hidden_channels)

        # output layer
        self.classfier = nn.Linear(hidden_channels, 2)

        self.mask_target = nn.parameter.Parameter(nn.init.xavier_uniform_(
                torch.empty(1, target_features_num)).float())
        self.mask_drug = nn.parameter.Parameter(nn.init.xavier_uniform_(
                torch.empty(1, drug_features_num)).float())

    def forward(self, drug1_id, drug2_id, cell_features, x_dict,egde_index_dict):
        x_dict = self.gnn(x_dict,egde_index_dict)

        # get drugs hidden_state
        drug1 = x_dict['drug'][drug1_id]
        drug2 = x_dict['drug'][drug2_id]
        # get cell hidden_state
        cell = self.reduction(cell_features)

        hidden = torch.cat((drug1, drug2, cell), 1)

        hidden = self.reduction2(hidden)

        output = self.classfier(hidden)

        output=F.softmax(output,dim=1)

        return output, x_dict

    def get_mask(self):
        return self.mask_drug,self.mask_target
