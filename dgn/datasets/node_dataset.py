import torch
from torch import  tensor
from torch_geometric.data import Data, Dataset
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T


def create_data(drug_feature, target_feature, dd_edge_index, dt_edge_index, tt_edge_index, dd_edge_att):

    data = HeteroData()

    data['drug'].x = drug_feature  # [num_papers, num_features_paper]
    data['target'].x = target_feature  # [num_authors, num_features_author]

    data['drug', 'd-d', 'drug'].edge_index = dd_edge_index  # [2, num_edges_cites]
    data['drug', 'd-t', 'target'].edge_index = dt_edge_index  # [2, num_edges_writes]
    # data['target', 't-d', 'drug'].edge_index = dt_edge_index[::-1]  # [2, num_edges_affiliated]
    data['target', 't-t', 'target'].edge_index = tt_edge_index  # [2, num_edges_topic]

    data['drug', 'd-d', 'drug'].edge_attr = dd_edge_att  # [num_edges_cites, num_features_d-d]
    data['drug', 'd-t', 'target'].edge_attr = torch.ones([dt_edge_index.shape[-1],1])  # [num_edges_writes, num_features_d-t]
    # data['target', 't-d', 'drug'].edge_attr = torch.ones([dt_edge_index.shape[-1],1])
    data['target', 't-t', 'target'].edge_attr = torch.ones([tt_edge_index.shape[-1],1])

    data = T.ToUndirected()(data)

    return data


# class GNNDataset(Dataset):
