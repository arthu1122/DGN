import math

import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

from utils.normalize import preprocess_adj, preprocess_features


class GATConv(nn.Module):
    def __init__(self, src_dst_feature, out_feature, alpha=0.02, dropout=0.1, concat=True):
        super(GATConv, self).__init__()

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

        self.att = nn.Parameter(nn.init.xavier_uniform_(torch.empty(2 * self.out_feature, 1)).float())

        self.alpha = alpha
        self.leakyrelu = nn.LeakyReLU(self.alpha)

        self.concat = concat

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, adj):
        """
        :param x: node features [ node_num × feature_num ] (tuple optional)
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
            dst_h = src_h
        else:
            if isinstance(x, tuple):
                src, dst = x
            else:
                raise TypeError("x must be a tuple")
            src_h = torch.mm(src, self.weight1)
            dst_h = torch.mm(dst, self.weight2)

        n1 = src_h.size()[0]
        n2 = dst_h.size()[0]

        # # get attention score
        # score = self.attention_score(dst_h, src_h)

        # [n1, n2, 2*out_features]
        a_input = torch.cat([src_h.repeat(1, n2).view(n1 * n2, -1), dst_h.repeat(n1, 1).view(n1 * n2, -1)], dim=1).view(n1, n2, 2 * self.out_feature)
        # [n1,n2]
        e = self.leakyrelu(torch.matmul(a_input, self.att).squeeze(2))
        zero_vec = -1e12 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = self.dropout(attention)
        # h_prime = torch.matmul(attention, h)

        h_prime = torch.matmul(attention.T, src_h)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def attention_score(self, q, k):
        d = q.shape[-1]
        scores = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(d)
        return F.softmax(scores, dim=-1)


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

    gat = GATConv((500, 800), 768)

    res = gat((x1, x2), adj)
    print(res)
