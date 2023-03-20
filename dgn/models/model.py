import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GATConv, GCNConv


class HeteroGNNs(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.num_layers = args.num_layers

        # graph network
        # self.convs = nn.ModuleList()

        self.tt = GATConv(args.target_features_num, args.hidden_channels, add_self_loops=False)
        self.dt = GATConv((args.drug_features_num, args.target_features_num), args.hidden_channels, add_self_loops=False)
        self.dd = GATConv(args.drug_features_num, args.hidden_channels, add_self_loops=False)

        self.td = GATConv((args.hidden_channels, args.drug_features_num), args.hidden_channels, add_self_loops=False)

        # for _ in range(self.num_layers):
        #     if args.gnn == 'gat':
        #         conv = HeteroConv({
        #
        #             ('target', 't-t', 'target'): GATConv(args.target_features_num, args.hidden_channels, add_self_loops=False),
        #             ('drug', 'd-t', 'target'): GATConv((args.drug_features_num, args.target_features_num), args.hidden_channels, add_self_loops=False),
        #             ('drug', 'd-d', 'drug'): GATConv(args.drug_features_num, args.hidden_channels, add_self_loops=False),
        #             ('target', 'rev_d-t', 'drug'): GATConv((args.target_features_num, args.drug_features_num), args.hidden_channels, add_self_loops=False),
        #
        #         }, aggr='sum')
        #
        #     else:
        #         raise NotImplementedError("GNN type not found")

            # self.convs.append(conv)

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        # for conv in self.convs:
        #     x_dict = conv(x_dict, edge_index_dict, edge_attr_dict)
        #     x_dict = {key: x.relu() for key, x in x_dict.items()}

        x_tt = self.tt((x_dict['target'], x_dict['target']), edge_index_dict[('target', 't-t', 'target')])
        x_dt = self.dt((x_dict['drug'], x_dict['target']), edge_index_dict[('drug', 'd-t', 'target')])

        x_target = (x_tt + x_dt) / 2
        x_dict['target'] = x_target

        x_dd = self.dd((x_dict['drug'], x_dict['drug']), edge_index_dict[('drug', 'd-d', 'drug')])
        x_td = self.td((x_dict['target'], x_dict['drug']), edge_index_dict[('target', 'rev_d-t', 'drug')])

        x_drug = (x_dd + x_td) / 2
        x_dict['drug'] = x_drug

        return x_dict


class Reduction(nn.Module):
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

        # gnn
        self.gnn = HeteroGNNs(args)
        # self.gnn.reset_params()

        # get cell features
        self.reduction = Reduction(args.cell_features_num, args.hidden_channels * 2, args)

        # final transform
        self.reduction2 = Reduction(args.hidden_channels * 4, args.hidden_channels, args)

        # output layer
        self.classfier = nn.Linear(args.hidden_channels, 2)

        if args.setting != 1:
            self.mask_target = nn.Parameter(nn.init.xavier_uniform_(torch.empty(1, args.target_features_num)).float())
            self.mask_drug = nn.Parameter(nn.init.xavier_uniform_(torch.empty(1, args.drug_features_num)).float())

    def forward(self, drug1_id, drug2_id, cell_features, x_dict, edge_index_dict, edge_attr_dict):
        x_dict = self.gnn(x_dict, edge_index_dict, edge_attr_dict)

        # get drugs hidden_state
        drug1 = x_dict['drug'][drug1_id]
        drug2 = x_dict['drug'][drug2_id]
        # get cell hidden_state
        cell = self.reduction(cell_features)

        hidden = torch.cat((drug1, drug2, cell), 1)

        hidden = self.reduction2(hidden)

        output = self.classfier(hidden)

        return output, x_dict

    def get_mask(self):
        return self.mask_drug, self.mask_target
