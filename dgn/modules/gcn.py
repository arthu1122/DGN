import torch.nn as nn
import torch


class GCNConv(nn.Module):
    def __init__(self, in_feature, out_feature, bias=True):
        super(GCNConv, self).__init__()

        self.in_feature = in_feature
        self.out_feature = out_feature

        self.weight = nn.Parameter(nn.init.xavier_uniform_(torch.empty(in_feature, out_feature)).float())

        if bias:
            self.bias = nn.Parameter(nn.init.xavier_uniform_(torch.empty(out_feature)).float())

    def forward(self, x, adj):
        """
        :param x: node features [ node_num × feature_num ]
        :param adj: adjacency matrix, a sparse tensor [ node_num × node_num ]
        :return: updated node features
        """
        support = torch.mm(x, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output
