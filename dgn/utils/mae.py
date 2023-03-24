import random

import numpy as np
import torch
import torch.nn.functional as F


def get_mask_index(total, ratio):
    """
    get mask index matrix

    :param total: num of nodes
    :param ratio: mask ratio
    :return: mask index matrix
    """
    mask = torch.zeros(total)
    select = random.sample(range(total), int(total * ratio))
    mask[select] = 1
    return mask.unsqueeze(-1), select


def get_mae_loss(x_dict, _x_dict, drug_mask_index, target_mask_index):
    """
    mae loss: cosine error in masked nodes

    :param x_dict: origin x_dict
    :param _x_dict: masked x_dict
    :param drug_mask_index: drug mask position
    :param target_mask_index: target mask position
    :return: cosine error in masked nodes
    """

    # get node features
    drug_features = x_dict['drug']
    target_features = x_dict['target']

    # leave the nodes masked
    drug_features_m = drug_features[drug_mask_index]
    target_features_m = target_features[target_mask_index]

    # as same as x_dict
    _drug_features = _x_dict['drug']
    _target_features = _x_dict['target']
    _drug_features_m = _drug_features[drug_mask_index]
    _target_features_m = _target_features[target_mask_index]

    def sce_loss(x, y, alpha=1):
        x = F.normalize(x, p=2, dim=-1)
        y = F.normalize(y, p=2, dim=-1)

        loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)

        loss = loss.mean()
        return loss

    drug_loss = sce_loss(_drug_features_m, drug_features_m)
    target_loss = sce_loss(_target_features_m, target_features_m)

    return drug_loss + target_loss


def get_mask_x_dict(x_dict, m_drug, m_target, ratio=0.1):
    """
    get masked x_dict

    :param x_dict: origin x_dict
    :param m_drug: learnable mask matrix
    :param m_target: learnable mask matrix
    :param ratio: mask ratio
    :return: masked x_dict
    """
    drug_features = x_dict['drug']
    target_features = x_dict['target']

    drug_mask, drug_mask_index = get_mask_index(drug_features.shape[0], ratio)
    target_mask, target_mask_index = get_mask_index(target_features.shape[0], ratio)

    m_drug = m_drug.to(drug_features.device)
    m_target = m_target.to(target_features.device)
    drug_mask = drug_mask.to(drug_features.device)
    target_mask = target_mask.to(target_features.device)

    drug_m = drug_mask * m_drug
    _drug_features = torch.mul(drug_features, torch.ones(drug_mask.size()).to(drug_features.device) - drug_mask) + drug_m

    target_m = target_mask * m_target
    _target_features = torch.mul(target_features, torch.ones(target_mask.size()).to(target_features.device) - target_mask) + target_m

    _x_dict = {'drug': _drug_features, 'target': _target_features}

    return _x_dict, drug_mask_index, target_mask_index,
