import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import HeteroConv, GATConv, Linear


class HeteroGNN(torch.nn.Module):
    def __init__(self, num_features_cell=890, hidden_channels=768, num_layers=1, dropout=0.2):
        super().__init__()

        # graph network
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                ('drug', 'd-d', 'drug'): GATConv((-1, -1), hidden_channels),
                ('drug', 'd-t', 'target'): GATConv((-1, -1), hidden_channels,add_self_loops=False),
                ('target', 't-t', 'target'): GATConv((-1, -1), hidden_channels,),
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
            nn.Linear(hidden_channels*4, 2048),
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


    def forward(self, drug1_id, drug2_id, cell_features, graph_data):

        x_dict = graph_data.collect('x')
        edge_index_dict = graph_data.collect('edge_index')
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: x.relu() for key, x in x_dict.items()}

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

        return output
