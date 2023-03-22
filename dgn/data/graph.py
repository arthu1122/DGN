import numpy as np
import scipy.sparse as sp
import torch


def build_adj(edges, num_node, edge_attr=None):
    """
    :param edge_attr: np [edge_num]
    :param edges: [edge_num × 2]
    :param num_node: if one type node 'int', else 'tuple'
    :return:
    """
    if isinstance(num_node, tuple):
        n1, n2 = num_node[0], num_node[1]
    elif isinstance(num_node, int):
        n1 = n2 = num_node
    else:
        raise TypeError("num_node must be int or tuple")

    edge_base = np.ones(edges.shape[0])

    adj = sp.coo_matrix((edge_base, (edges[:, 0], edges[:, 1])),
                        shape=(n1, n2),
                        dtype=np.float32)
    if edge_attr is not None:
        edge_attr = sp.coo_matrix((edge_attr, (edges[:, 0], edges[:, 1])),
                                  shape=(n1, n2),
                                  dtype=np.float32)

    return adj, edge_attr


def normalize_adj(adj):
    """
    Symmetrically normalize adjacency matrix.
    """
    col_sum = np.array(adj.sum(0))
    row_sum = np.array(adj.sum(1))
    c_inv = np.power(col_sum, -0.5).flatten()
    r_inv = np.power(row_sum, -0.5).flatten()
    c_inv[np.isinf(c_inv)] = 0.
    r_inv[np.isinf(r_inv)] = 0.
    c_mat_inv = sp.diags(c_inv)
    r_mat_inv = sp.diags(r_inv)
    return adj.dot(c_mat_inv).transpose().dot(r_mat_inv).transpose().tocoo()


def preprocess_adj(edges, num_node, edge_attr=None, undirected=False, add_self_loop=False, normalize=False):
    """
    adj preprocess in GCN
    :param edge_attr:
    :param normalize:
    :param add_self_loop:
    :param undirected: if one type node and undirected, set True
    :param num_node: tuple or int
    :param edges:  [edge_num × 2]
        example:[[1,2],[1,3],[1,4] ... ]
    :return:
    """
    adj, attr = build_adj(edges, num_node, edge_attr)

    if undirected:
        # build symmetric adjacency matrix
        assert isinstance(num_node, int), "only one type node can set undirected"
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    if add_self_loop:
        adj = adj + sp.eye(adj.shape[0])
        if attr is not None:
            attr=attr+sp.eye(attr.shape[0])

    if normalize:
        adj = normalize_adj(adj)

    adj = torch.FloatTensor(np.array(adj.todense()))
    if attr is not None:
        attr = torch.FloatTensor(np.array(attr.todense()))

    return adj, attr


def normalize_features(mx):
    """
    Row-normalize sparse matrix
    """
    row_sum = np.array(mx.sum(1))
    r_inv = np.power(row_sum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def preprocess_features(features):
    """
    normalized features
    :param features:
    :return:
    """
    features = sp.csr_matrix(features, dtype=np.float32)
    features_normalized = normalize_features(features)
    return torch.FloatTensor(np.array(features_normalized.todense()))


class GraphData:
    def __init__(self, node_dict, adj_dict, edge_attr_dict):
        self._node_dict = node_dict
        self._adj_dict = adj_dict
        self._edge_attr_dict = edge_attr_dict

    @property
    def node_dict(self):
        return self._node_dict

    @property
    def adj_dict(self):
        return self._adj_dict

    @property
    def edge_attr_dict(self):
        return self._edge_attr_dict
