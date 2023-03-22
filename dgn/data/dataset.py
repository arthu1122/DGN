import torch
from torch.utils.data import Dataset


class GNNDataset(Dataset):
    def __init__(self, label_df, ID2id, features_cell_df, device):
        super(GNNDataset, self).__init__()

        # # TODO
        # label_df=label_df[:100]

        cell_list = list(label_df['Cell_Line_Name'].str.lower())
        drug1_list = list(label_df['DrugID1'])
        drug2_list = list(label_df['DrugID2'])
        label_list = list(label_df['Label'])

        features_cell_df.index = features_cell_df.index.str.lower()

        features_cell_np = features_cell_df.loc[cell_list].to_numpy().astype('float')
        self.features_cell_t = torch.FloatTensor(features_cell_np) \
            # .to(device)
        self.labels_t = torch.LongTensor(label_list) \
            # .to(device)

        self.ID2id = ID2id
        drug1_id_list = [self.ID2id["Drug|||" + str(x)] for x in drug1_list]
        drug2_id_list = [self.ID2id["Drug|||" + str(x)] for x in drug2_list]

        self.drug1_t = torch.LongTensor(drug1_id_list) \
            # .to(device)
        self.drug2_t = torch.LongTensor(drug2_id_list) \
            # .to(device)

    def __getitem__(self, index):
        return self.drug1_t[index].unsqueeze(-1), self.drug2_t[index].unsqueeze(-1), self.features_cell_t[index], self.labels_t[index].unsqueeze(-1)

    def __len__(self):
        return len(self.labels_t)
