import numpy as np
import torch.nn as nn
import torch

from utils.normalize import preprocess_adj, preprocess_features


class GCNConv(nn.Module):
    def __init__(self, src_dst_feature, out_feature, bias=True):
        super(GCNConv, self).__init__()

        if isinstance(src_dst_feature, int):
            self.src_feature = src_dst_feature
            self.dst_feature = self.src_feature
        elif isinstance(src_dst_feature, tuple):
            self.src_feature, self.dst_feature = src_dst_feature
        else:
            raise TypeError("src_dst_feature must be int or tuple")

        self.out_feature = out_feature

        self.weight1 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.src_feature, self.out_feature)).float())
        if isinstance(src_dst_feature, tuple):
            self.weight2 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.dst_feature, self.out_feature)).float())
        else:
            self.weight2 = None

        self.bias = bias
        if self.bias:
            self.bias1 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(1, out_feature)).float())
            if isinstance(src_dst_feature, tuple):
                self.bias2 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(1, self.out_feature)).float())
            else:
                self.bias2 = None

    def forward(self, x, adj):
        """
        :param x: node features [ node_num × feature_num ]
        :param adj: normalized adjacency matrix, a sparse tensor [ node_num × node_num ]
        :return: updated node features
        """
        if self.weight2 is None:
            if isinstance(x, tuple):
                src, dst = x
                assert src == dst, "src != dst"
            elif isinstance(x, torch.Tensor):
                src = x
            else:
                raise TypeError("x must be a tuple or tensor")
            src_h = torch.mm(src, self.weight1)
            aggr_src = torch.spmm(adj.T, src_h)
            if self.bias:
                return aggr_src + self.bias1
        else:
            if isinstance(x, tuple):
                src, dst = x
            else:
                raise TypeError("x must be a tuple")
            src_h = torch.mm(src, self.weight1)
            dst_h = torch.mm(dst, self.weight2)
            aggr_dst = torch.spmm(adj, dst_h)
            aggr_src = torch.spmm(adj.T, src_h)
            if self.bias:
                return aggr_src + self.bias1, aggr_dst + self.bias2


if __name__ == '__main__':
    x1 = np.random.random((60, 500))
    x2 = np.random.random((120, 800))

    x2index = np.arange(120)[np.newaxis, :]
    x1index = np.random.randint(0, 59, 120)[np.newaxis, :]

    edges = np.concatenate((x1index, x2index), axis=0)
    edges = edges.T

    adj = preprocess_adj(edges, (60, 120))
    x1 = preprocess_features(x1)
    x2 = preprocess_features(x2)

    gcn = GCNConv((500, 800), 768)

    res = gcn((x1, x2), adj)
    print(res)
