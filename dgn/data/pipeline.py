import torch

from data.graph import preprocess_adj, GraphData


class GraphPipeline():
    @staticmethod
    def get_data(features_drug, features_target, e_index_dd, e_index_dt, e_index_tt, e_attr_dd, device):
        # --------------- Basic Data Prepare ----------------------------------------
        attr_dd = torch.FloatTensor([float(attr[:-1]) for attr in e_attr_dd])

        drug_feature = torch.FloatTensor(features_drug.values.astype('float'))
        target_feature = torch.FloatTensor(features_target.values.astype('float'))

        def transform(indexs):
            # "[0,1]\n" -> [0,1]
            begin = []
            end = []
            for index in indexs:
                begin.append(int(index[1:-2].split(",")[0]))
                end.append(int(index[1:-2].split(",")[1]))
            return torch.tensor([begin, end])

        edge_index_dd = transform(e_index_dd)
        edge_index_dt = transform(e_index_dt)
        edge_index_tt = transform(e_index_tt)

        # --------------- Adj Build  ----------------------------------------
        adj_dd, attr_dd = preprocess_adj(edge_index_dd.transpose(0, 1), drug_feature.shape[0], attr_dd, undirected=True, add_self_loop=True)
        adj_dt, _ = preprocess_adj(edge_index_dt.transpose(0, 1), (drug_feature.shape[0], target_feature.shape[0]))
        adj_tt, _ = preprocess_adj(edge_index_tt.transpose(0, 1), target_feature.shape[0], undirected=True, add_self_loop=True)

        mask_dd = torch.eye(adj_dd.shape[0])
        mask_dt = torch.zeros_like(adj_dt)
        mask_tt = torch.eye(adj_tt.shape[0])

        def combine(adj1, adj2, adj3):
            t1 = torch.concat((adj1, adj2), -1)
            t2 = torch.concat((adj2.transpose(0, 1), adj3), -1)
            return torch.concat((t1, t2), 0).unsqueeze(0)

        only_dd = 1 - combine(adj_dd, mask_dt, mask_tt)
        only_dt = 1 - combine(mask_dd, adj_dt, mask_tt)
        only_tt = 1 - combine(mask_dd, mask_dt, adj_tt)
        mask=torch.concat((only_dd,only_dt,only_tt),0)


        only_dd_attr = combine(attr_dd, mask_dt, mask_tt)

        # --------------- Structure Data Return  ----------------------------------------
        node_dict = {
            "drug": drug_feature.to(device),
            "target": target_feature.to(device)
        }

        adj_dict = {
            "dd": adj_dd.to(device),
            "dt": adj_dt.to(device),
            "tt": adj_tt.to(device),
        }


        attr_dict = {
            "dd": attr_dd.to(device),
            "only_dd": only_dd_attr.to(device)
        }

        return GraphData(node_dict, adj_dict, attr_dict,mask)
