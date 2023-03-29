import torch
import torch.nn as nn
import torch.nn.functional as F

from data.graph import GraphData
from modules.gat import TRMGATLayer


class GNN(nn.Module):
    def __init__(self, args):
        super(GNN, self).__init__()

        self.layers = nn.ModuleList()
        self.num_layers = args.num_layers
        for _ in range(self.num_layers):
            self.layers.append(TRMGATLayer(args.hidden_channels, args.qk_dim, 3, args.dropout))

    def forward(self, all_nodes, mask, edge_attr_dict):
        for layer in self.layers:
            all_nodes = layer(all_nodes, mask, edge_attr_dict)

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
        return self.reduction(F.normalize(x, 2, 0))


class UnnamedModel(nn.Module):
    def __init__(self, args):
        super().__init__()

        # node to uniform dim
        self.fc_drug = nn.Linear(args.drug_features_num, args.hidden_channels)
        self.fc_target = nn.Linear(args.target_features_num, args.hidden_channels)

        self.fc_cell = FFN(args.cell_features_num, args.hidden_channels * 2, args)

        self.gnn = GNN(args)

        self.ffn = FFN(args.hidden_channels * 4, args.hidden_channels, args)

        # output layer
        self.classfier = nn.Linear(args.hidden_channels, 2)

        if args.setting not in [1, 5]:
            self.mask_target = nn.Parameter(nn.init.xavier_uniform_(torch.empty(1, args.target_features_num)).float())
            self.mask_drug = nn.Parameter(nn.init.xavier_uniform_(torch.empty(1, args.drug_features_num)).float())

    def forward(self, drug1_id, drug2_id, cell_features, graph_data):
        drug = graph_data.node_dict['drug']
        target = graph_data.node_dict['target']

        # 统一drug、target维度
        h_drug = self.fc_drug(drug)
        h_target = self.fc_target(target)
        all_nodes = torch.concat((h_drug, h_target), 0)

        # 图神经网络更新节点表示
        _h_nodes = self.gnn(all_nodes, graph_data.mask, graph_data.edge_attr_dict)
        _x_dict = {'drug': _h_nodes[:drug.shape[0]], 'target': _h_nodes[drug.shape[0]:]}

        # 得到两个药物表示
        h_drug1 = _x_dict['drug'][drug1_id]
        h_drug2 = _x_dict['drug'][drug2_id]

        # 得到细胞系表示
        h_cell = self.fc_cell(cell_features)

        hidden = torch.cat((h_drug1, h_drug2, h_cell), -1)
        hidden = self.ffn(hidden)

        output = self.classfier(hidden)
        return output, _x_dict

    def get_mask(self):
        return self.mask_drug, self.mask_target
