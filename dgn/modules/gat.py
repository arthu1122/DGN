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
        wx = self.weight1(x)
        a_input = self._prepare_attentional_mechanism_input(wx)
        e = self.leakyrelu(self.att(a_input).squeeze(2))

        if edge_attr is not None:
            e = e + edge_attr

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, wx)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        N = Wh.size()[0]

        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
        Wh_repeated_alternating = Wh.repeat(N, 1)
        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)
        return all_combinations_matrix.view(N, N, 2 * self.out_feature)


class TRMGATLayer(nn.Module):
    def __init__(self, d_model, qk_dim, head, dropout=0.1):
        super(TRMGATLayer, self).__init__()

        self.in_feature = d_model

        self.out_feature = d_model
        self.qk_dim = qk_dim

        self.head = head

        self.q_transform = nn.Linear(self.in_feature, self.qk_dim*self.head)
        self.k_transform = nn.Linear(self.in_feature, self.qk_dim*self.head)
        self.v_transform = nn.Linear(self.in_feature, self.out_feature*self.head)

        self.out = nn.Linear(self.out_feature*self.head, self.out_feature)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask, edge_attr_dict=None):
        """
        :param x: node features [ node_num × feature_num ]
        :return: updated node features
        """

        q = self.q_transform(x).view((-1, self.head, self.qk_dim )).transpose(-2, -3)
        k = self.k_transform(x).view((-1, self.head, self.qk_dim )).transpose(-2, -3)
        v = self.v_transform(x).view((-1, self.head, self.out_feature)).transpose(-2, -3)

        d = k.shape[-1]
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d)

        if edge_attr_dict is not None:
            scores[0] = scores[0] + edge_attr_dict['only_dd']

        scores.masked_fill_(mask, -1e9)

        attention = torch.softmax(scores, dim=-1)
        attention = self.dropout(attention)

        result = torch.matmul(attention, v)

        result = result.transpose(-2, -3).reshape(-1, self.out_feature*self.head)

        output = self.out(result)

        return output
