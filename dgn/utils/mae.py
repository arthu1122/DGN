import random

import numpy as np
import torch


def get_mask_index(total, ratio):
    ranks = np.arange(total)
    sample_num = int(ratio * total)
    indices = random.sample(list(ranks), sample_num)
    mask = torch.zeros((total, 1))
    mask[indices] = 1
    return mask


def get_mae_loss(x_dict, _x_dict, drug_mask_index, target_mask_index):
    drug_features = x_dict['drug']
    drug_features_m = torch.mul(drug_features, drug_mask_index)
    target_features = x_dict['target']
    target_features_m = torch.mul(target_features, target_mask_index)

    _drug_features = _x_dict['drug']
    _drug_features_m = torch.mul(_drug_features, drug_mask_index)
    _target_features = _x_dict['target']
    _target_features_m = torch.mul(_target_features, target_mask_index)

    drug_features_num = drug_features.shape[0]
    drug_loss = torch.sum(
        torch.ones(drug_features_num).to(drug_features.device) - torch.cosine_similarity(_drug_features_m,
                                                                                         drug_features_m,
                                                                                         dim=1)) / drug_features_num
    target_features_num = target_features.shape[0]
    target_loss = torch.sum(
        torch.ones(target_features_num).to(target_features.device) - torch.cosine_similarity(_target_features_m,
                                                                                             target_features_m,
                                                                                             dim=1)) / target_features_num

    return drug_loss + target_loss
