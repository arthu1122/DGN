import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GATConv

from dgn.utils.mae import get_mae_loss, get_mask_index


class HeteroGNNs(nn.Module):
    def __init__(self, num_layers=1, hidden_channels=768, target_features_num=570, drug_features_num=200, mae=True):
        super().__init__()
        self.mae = mae

        if self.mae:
            self.mask_target = nn.parameter.Parameter(nn.init.xavier_uniform_(
                torch.empty(1, target_features_num)).float())
            self.mask_drug = nn.parameter.Parameter(nn.init.xavier_uniform_(
                torch.empty(1, drug_features_num)).float())

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

    def forward(self, graph_data, ratio=0.2):

        x_dict = graph_data.collect('x')
        edge_index_dict = graph_data.collect('edge_index')
        drug_features = x_dict['drug']
        target_features = x_dict['target']

        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: x.relu() for key, x in x_dict.items()}

        loss_mae = None

        if self.mae:
            drug_mask_index = get_mask_index(drug_features.shape[0], ratio)
            target_mask_index = get_mask_index(target_features.shape[0], ratio)

            self.mask_drug = self.mask_drug.to(drug_features.device)
            drug_mask_index = drug_mask_index.to(drug_features.device)
            self.mask_target = self.mask_target.to(target_features.device)
            target_mask_index = target_mask_index.to(target_features.device)

            drug_m = drug_mask_index * self.mask_drug
            _drug_features = torch.mul(drug_features, torch.ones(drug_mask_index.size()).to(drug_features.device) - drug_mask_index) + drug_m

            target_m = target_mask_index * self.mask_target
            _target_features = torch.mul(target_features,
                                         torch.ones(target_mask_index.size()).to(target_features.device) - target_mask_index) + target_m

            _x_dict = {'drug': _drug_features, 'target': _target_features}

            for conv in self.convs:
                _x_dict = conv(_x_dict, edge_index_dict)
                _x_dict = {key: x.relu() for key, x in _x_dict.items()}

            loss_mae = get_mae_loss(x_dict, _x_dict, drug_mask_index, target_mask_index)

        return x_dict, loss_mae


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
    def __init__(self, num_features_cell=890, hidden_channels=768, dropout=0.2):
        super().__init__()

        self.gnns = HeteroGNNs()

        # get cell features
        self.reduction = Reduction(num_features_cell, hidden_channels * 2)

        # final transform
        self.reduction2 = Reduction(hidden_channels * 4, hidden_channels)

        # output layer
        self.classfier = nn.Linear(hidden_channels, 2)

    def forward(self, drug1_id, drug2_id, cell_features, graph_data):
        x_dict, loss_mae = self.gnns(graph_data)

        # get drugs hidden_state
        drug1 = x_dict['drug'][drug1_id]
        drug2 = x_dict['drug'][drug2_id]
        # get cell hidden_state
        cell = self.reduction(cell_features)

        hidden = torch.cat((drug1, drug2, cell), 1)

        hidden = self.reduction2(hidden)

        output = self.classfier(hidden)

        return output, loss_mae
