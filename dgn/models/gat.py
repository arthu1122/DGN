import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import HeteroConv, GATConv, Linear
import numpy as np
import random


class HeteroGNN(torch.nn.Module):
    def __init__(self, num_features_cell=890, hidden_channels=768, num_layers=1, dropout=0.2, mae=True,
                 drug_features_num=200, target_features_num=570):
        super().__init__()

        self.mae = mae

        if self.mae:
            self.mask_target = nn.parameter.Parameter(nn.init.xavier_uniform_(
                torch.empty(1, target_features_num)).float(), requires_grad=True)
            self.mask_drug = nn.parameter.Parameter(nn.init.xavier_uniform_(
                torch.empty(1, drug_features_num)).float(), requires_grad=True)

        # graph network
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                ('drug', 'd-d', 'drug'): GATConv((-1, -1), hidden_channels),
                ('drug', 'd-t', 'target'): GATConv((-1, -1), hidden_channels, add_self_loops=False),
                ('target', 'rev_d-t', 'drug'): GATConv((-1, -1), hidden_channels, add_self_loops=False),
                ('target', 't-t', 'target'): GATConv((-1, -1), hidden_channels, ),
            }, aggr='sum')
            self.convs.append(conv)

        # get cell features
        self.reduction = nn.Sequential(
            nn.Linear(num_features_cell, 2048),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, hidden_channels * 2),
            nn.ReLU()
        )

        # final transform
        self.reduction2 = nn.Sequential(
            nn.Linear(hidden_channels * 4, 2048),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, hidden_channels),
            nn.ReLU()
        )

        # output layer
        self.classfier = nn.Linear(hidden_channels, 2)

    def forward(self, drug1_id, drug2_id, cell_features, graph_data, ratio=0.2):

        x_dict = graph_data.collect('x')
        edge_index_dict = graph_data.collect('edge_index')
        drug_features = x_dict['drug']
        target_features = x_dict['target']

        if self.mae:
            drug_mask_index = self.get_mask_index(drug_features.shape[0], ratio)
            target_mask_index = self.get_mask_index(target_features.shape[0], ratio)

            self.mask_drug = self.mask_drug.to(drug_features.device)
            drug_mask_index = drug_mask_index.to(drug_features.device)
            self.mask_target = self.mask_target.to(target_features.device)
            target_mask_index = target_mask_index.to(target_features.device)

            drug_m = drug_mask_index * self.mask_drug
            _drug_features = torch.mul(drug_features, torch.ones(drug_mask_index.size()) - drug_mask_index) + drug_m

            target_m = target_mask_index * self.mask_target
            _target_features = torch.mul(target_features,
                                         torch.ones(target_mask_index.size()) - target_mask_index) + target_m

            _x_dict = {'drug': _drug_features, 'target': _target_features}

            for conv in self.convs:
                _x_dict = conv(_x_dict, edge_index_dict)
                _x_dict = {key: x.relu() for key, x in _x_dict.items()}

        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: x.relu() for key, x in x_dict.items()}

        if self.mae:
            loss_mae = self.get_mae_loss(x_dict, _x_dict, drug_mask_index, target_mask_index)

        # get drugs hidden_state
        drug1 = x_dict['drug'][drug1_id]
        drug2 = x_dict['drug'][drug2_id]
        # get cell hidden_state
        cell_features = F.normalize(cell_features, 2, 1)
        cell = self.reduction(cell_features)

        hidden = torch.cat((drug1, drug2, cell), 1)
        hidden = F.normalize(hidden, 2, 1)
        hidden = self.reduction2(hidden)

        output = self.classfier(hidden)

        return output, loss_mae

    def get_mask_index(self, total, ratio):
        ranks = np.arange(total)
        sample_num = int(ratio * total)
        indices = random.sample(list(ranks), sample_num)
        mask = torch.zeros((total, 1))
        mask[indices] = 1
        return mask

    def get_mae_loss(self, x_dict, _x_dict, drug_mask_index, target_mask_index):
        drug_features = x_dict['drug']
        drug_features_m = torch.mul(drug_features, drug_mask_index)
        target_features = x_dict['target']
        target_features_m = torch.mul(target_features, target_mask_index)

        _drug_features = _x_dict['drug']
        _drug_features_m = torch.mul(_drug_features, drug_mask_index)
        _target_features = _x_dict['target']
        _target_features_m = torch.mul(_target_features, target_mask_index)

        drug_loss = torch.sum(
            torch.ones(drug_features.shape[0]) - torch.cosine_similarity(_drug_features_m, drug_features_m, dim=1)) / \
                    drug_features.shape[0]

        target_loss = torch.sum(
            torch.ones(target_features.shape[0]) - torch.cosine_similarity(_target_features_m, target_features_m, dim=1)) / \
                    target_features.shape[0]

        return drug_loss + target_loss
