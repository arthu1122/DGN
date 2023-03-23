import math

import torch.nn as nn
import torch
import torch.nn.functional as F


class GATLayer(nn.Module):
    def __init__(self, in_feature, out_feature, alpha=0.02, dropout=0.1, concat=True):
        super(GATLayer, self).__init__()

        self.in_feature = in_feature
        self.out_feature = out_feature

        self.weight1 = nn.Linear(self.in_feature, self.out_feature)
        self.att = nn.Linear(self.out_feature * 2, 1)

        self.alpha = alpha
        self.leakyrelu = nn.LeakyReLU(self.alpha)

        self.concat = concat
        self.dropout = dropout

    def forward(self, x, adj, edge_attr=None):
        """
        :param edge_attr:
        :param x: node features [ node_num × feature_num ]
        :param adj: normalized adjacency matrix, a sparse tensor [ node_num × node_num ]
        :return: updated node features
        """
        Wh = self.weight1(x)
        a_input = self._prepare_attentional_mechanism_input(Wh)  # 每一个节点和所有节点，特征。(Vall, Vall, feature)
        e = self.leakyrelu(self.att(a_input).squeeze(2))

        if edge_attr is not None:
            e = e + edge_attr

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)  # 将邻接矩阵中小于0的变成负无穷
        attention = F.softmax(attention, dim=1)  # 按行求softmax。 sum(axis=1) === 1
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)  # 聚合邻居函数

        if self.concat:
            return F.elu(h_prime)  # elu-激活函数
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        N = Wh.size()[0]  # number of nodes

        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)  # 复制
        Wh_repeated_alternating = Wh.repeat(N, 1)
        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)
        return all_combinations_matrix.view(N, N, 2 * self.out_feature)


class TRMGATLayer(nn.Module):
    def __init__(self, in_feature, out_feature, dropout=0.1, concat=True):
        super(TRMGATLayer, self).__init__()

        self.in_feature = in_feature

        self.out_feature = out_feature

        self.q_transform = nn.Linear(self.in_feature, self.out_feature)
        self.k_transform = nn.Linear(self.in_feature, self.out_feature)
        self.v_transform = nn.Linear(self.in_feature, self.out_feature)

        self.concat = concat

        self.dropout = dropout

    def forward(self, x, adj, edge_attr=None):
        """
        :param edge_attr:
        :param x: node features [ node_num × feature_num ]
        :param adj: normalized adjacency matrix, a sparse tensor [ node_num × node_num ]
        :return: updated node features
        """
        q = self.q_transform(x)
        k = self.k_transform(x)
        v = self.v_transform(x)

        d = q.shape[-1]
        scores = torch.matmul(q, k.transpose(0, 1)) / math.sqrt(d)

        if edge_attr is not None:
            scores = scores + edge_attr

        zero_vec = -9e15 * torch.ones_like(scores)
        scores = torch.where(adj > 0, scores, zero_vec)

        attention = torch.softmax(scores, dim=-1)
        attention = F.dropout(attention, self.dropout, training=self.training)

        att = torch.matmul(attention, v)

        if self.concat:
            return F.elu(att)  # elu-激活函数
        else:
            return att
