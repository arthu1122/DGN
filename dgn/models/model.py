import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.gat import GATLayer, TRMGATLayer


class GNN(nn.Module):
    def __init__(self, args):
        super(GNN, self).__init__()

        self.gnns = nn.ModuleList()
        self.num_layers = args.num_layers
        for _ in range(self.num_layers):
            self.gnns.append(TRMGATLayer(args.hidden_channels*3, args.qk_dim*3, 3))

    def forward(self, all_nodes, mask, edge_attr_dict):

        for gnn in self.gnns:
            all_nodes = gnn(all_nodes, mask, edge_attr_dict)

        return all_nodes


class FFN(nn.Module):
    def __init__(self, input_size, output_size, args):
        super().__init__()
        self.reduction = nn.Sequential(
            nn.Linear(input_size, args.project1),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(args.project1, args.project2),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(args.project2, output_size),
            nn.ReLU()
        )

    def forward(self, x):
        return self.reduction(F.normalize(x, 2, 1))


class UnnamedModel(nn.Module):
    def __init__(self, args):
        super().__init__()

        # node to uniform dim
        self.drug_fc = nn.Linear(args.drug_features_num, args.hidden_channels*3)
        self.target_fc = nn.Linear(args.target_features_num, args.hidden_channels*3)

        # gnn
        self.gnn = GNN(args)

        # get cell features
        self.ffn1 = FFN(args.cell_features_num, args.hidden_channels * 2, args)

        # final transform
        self.ffn2 = FFN(args.hidden_channels * 4, args.hidden_channels, args)

        # output layer
        self.classfier = nn.Linear(args.hidden_channels, 2)

        if args.setting != 1:
            self.mask_target = nn.Parameter(nn.init.xavier_uniform_(torch.empty(1, args.target_features_num)).float())
            self.mask_drug = nn.Parameter(nn.init.xavier_uniform_(torch.empty(1, args.drug_features_num)).float())

    def forward(self, drug1_id, drug2_id, cell_features, x_dict, mask, edge_attr_dict):
        drug = x_dict['drug']
        target = x_dict['target']
        # To unified dim
        _drug = self.drug_fc(drug)
        _target = self.target_fc(target)
        all_nodes = torch.concat((_drug, _target), 0)

        # Get drug graph representation
        _all_nodes = self.gnn(all_nodes, mask, edge_attr_dict)
        _x_dict = {'drug': _all_nodes[:drug.shape[0]], 'target': _all_nodes[drug.shape[0]:]}
        drug1 = _all_nodes[drug1_id]
        drug2 = _all_nodes[drug2_id]

        # Get cell representation
        cell = self.ffn1(cell_features)

        # Combine
        hidden = torch.cat((drug1, drug2, cell), -1)
        hidden = self.ffn2(hidden)

        output = self.classfier(hidden)

        return output, _x_dict

    def get_mask(self):
        return self.mask_drug, self.mask_target
