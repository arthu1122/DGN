import torch.nn as nn
import torch
import torch.nn.functional as F


class GATConv(nn.Module):
    def __init__(self, src_dst_feature, out_feature,dropout=0.1, bias=True):
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
        if isinstance(src_dst_feature, int):
            self.weight2 = None
        elif isinstance(src_dst_feature, tuple):
            self.weight2 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.dst_feature, self.out_feature)).float())
        else:
            raise TypeError("src_dst_feature must be int or tuple")

        self.att = nn.Parameter(nn.init.xavier_uniform_(torch.empty(2 * self.out_feature, 1)).float())

        if bias:
            self.bias = nn.Parameter(nn.init.xavier_uniform_(torch.empty(out_feature)).float())

        self.dropout=nn.Dropout(p=dropout)

    def forward(self, x, adj):
        """
        :param x: node features [ node_num × feature_num ]
        :param adj: adjacency matrix, a sparse tensor [ node_num × node_num ]
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
        n2 = dst_h.size()[1]

        # [n1, n2, 2*out_features]
        a_input = torch.cat([src_h.repeat(1, n2).view(n1 * n2, -1), dst_h.repeat(n1, 1).view(n1 * n2, -1)], dim=1).view(n1, n2, 2 * self.out_features)
        # [n1,n2]
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        zero_vec = -1e12 * torch.ones_like(e)  # 将没有连接的边置为负无穷
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = self.dropout(attention) # dropout，防止过拟合
        # h_prime = torch.matmul(attention, h)


        support = torch.mm(x, self.weight)  # (2708, 16) = (2708, 1433) X (1433, 16)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output
