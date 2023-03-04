import argparse
import datetime
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
from sklearn.metrics import cohen_kappa_score, accuracy_score, roc_auc_score, precision_score, recall_score, \
    balanced_accuracy_score
from sklearn.metrics import confusion_matrix
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.dataset import GNNDataset, get_graph_data
from utils.ema import initializes_target_network, update_target_network_parameters
from utils.mae import get_mask_x_dict, get_mae_loss
from models.gat import UnnamedModel

Default_Hparams = {
    "train_batch_size": 128,
    "test_batch_size": 128,
    "lr": 1e-5,
    "epoch": 200,
    "device": 0,
    "log_step": 20,
    "data": "../data/processed/Shuffled_Label_Data.csv",
    "f_drug": "../data/raw/Feature_DRUG.csv",
    "f_target": "../data/raw/Feature_TAR.csv",
    "f_cell": "../data/raw/Feature_CELL.csv",
    "dd_edge": "../data/processed/drug_drug_edge_index.txt",
    "dt_edge": "../data/processed/drug_tar_edge_index.txt",
    "tt_edge": "../data/processed/tar_tar_edge_index.txt",
    "dd_att": "../data/processed/drug_drug_edge_att.txt",
    "drug_vocab": "../data/processed/drug_vocab.txt",
    "output": '../data/result/',
}


def get_hparams(args):
    parser = argparse.ArgumentParser(
        description="Train a neural machine translation model.",
        usage="trainer.py [<args>] [-h | --help]"
    )

    parser.add_argument("--batch_size", type=int, help=" batch_size")
    parser.add_argument("--lr", type=float, help="Path to pre-trained checkpoint.")
    parser.add_argument("--epoch", type=int, help="epoch for train")
    parser.add_argument("--device", type=int, help="device")
    parser.add_argument("--log_step", type=int, help="when to print log")
    parser.add_argument("--data", type=str, help="training data with labels(.csv)")
    parser.add_argument("--f_drug", type=str, help="drug features file(.csv)")
    parser.add_argument("--f_target", type=str, help="target features file(.csv)")
    parser.add_argument("--f_cell", type=str, help="cell features file(.csv)")
    parser.add_argument("--dd_edge", type=str, help="drug-drug edge index file(.txt)")
    parser.add_argument("--dt_edge", type=str, help="drug-target edge index file(.txt)")
    parser.add_argument("--tt_edge", type=str, help="target-target edge index file(.txt)")
    parser.add_argument("--dd_att", type=str, help="drug-drug edge attribute file(.txt)")
    parser.add_argument("--drug_vocab", type=str, help="drug encoded file(.txt)")
    parser.add_argument("--output", type=str, help="output file fold")

    parsed_args = parser.parse_args(args)
    params = Default_Hparams

    # override
    arg_dict = vars(parsed_args)
    for key, item in arg_dict.items():
        if key in Default_Hparams.keys() and item:
            params[key] = item

    timestr = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    os.mkdir(params['output'] + timestr)
    params['output'] = params['output'] + timestr + "/"

    return params


def predict(model, device, loader_test, graph_data, mae=True):
    model.to(device)
    model.eval()

    with torch.no_grad():
        total_preds = torch.Tensor()
        total_labels = torch.Tensor()
        total_prelabels = torch.Tensor()
        print('Make prediction for {} samples...'.format(len(loader_test.dataset)))

        for step, data in tqdm(enumerate(loader_test), desc='Dev Itreation:'):
            print("Dev Step[{}/{}]".format(step + 1, len(loader_test)))

            drug1_ids, drug2_ids, cell_features, labels = data
            cell_features = cell_features.to(device)
            labels = labels.to(device)

            x_dict = graph_data.collect('x')
            edge_index_dict = graph_data.collect('edge_index')

            output, _ = model(drug1_ids, drug2_ids, cell_features, x_dict, edge_index_dict)

            ys = F.softmax(output, 1).to('cpu').data.numpy()
            predicted_labels = list(map(lambda x: np.argmax(x), ys))
            predicted_scores = list(map(lambda x: x[1], ys))
            total_preds = torch.cat((total_preds, torch.Tensor(predicted_scores)), 0)
            total_prelabels = torch.cat((total_prelabels, torch.Tensor(predicted_labels)), 0)
            total_labels = torch.cat((total_labels, labels.cpu()), 0)

    return total_labels.numpy().flatten(), total_preds.numpy().flatten(), total_prelabels.numpy().flatten()


def train(device, graph_data, loader_train, loss_fn, online_model, optimizer, log_step, epoch, epochs,
          target_model=None, m=1.0):
    online_model.to(device)
    online_model.train()
    for batch_idx, data in enumerate(loader_train):
        drug1_ids, drug2_ids, cell_features, labels = data

        cell_features = cell_features.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        x_dict = graph_data.collect('x')
        edge_index_dict = graph_data.collect('edge_index')

        mask_drug, mask_target = online_model.get_mask()
        _x_dict, drug_mask_index, target_mask_index = get_mask_x_dict(x_dict, mask_drug, mask_target,
                                                                      ratio=0.1)

        _output, _x_dict = online_model(drug1_ids, drug2_ids, cell_features, _x_dict, edge_index_dict)

        output1, x_dict1 = online_model(drug1_ids, drug2_ids, cell_features, x_dict, edge_index_dict)

        output, x_dict = target_model(drug1_ids, drug2_ids, cell_features, x_dict, edge_index_dict)

        loss = loss_fn(_output, labels)
        loss1 = loss_fn(output1, labels)

        loss_mae = get_mae_loss(x_dict, _x_dict, drug_mask_index, target_mask_index)

        loss_kl = (output1 * (torch.log(F.softmax(output1, 1)) - torch.log(F.softmax(_output, 1)))).sum(dim=1).sum()

        loss = loss + loss1 + loss_kl + 0.1 * loss_mae

        loss.backward()
        optimizer.step()

        update_target_network_parameters(online_model, target_model, m)

        if batch_idx % log_step == 0:
            print("[Train] {} Epoch[{}/{}],step[{}/{}],loss={:.6f}".format(
                datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), epoch + 1, epochs, batch_idx + 1, len(loader_train), loss.item()))


# get batch
def batch_collate(batch):
    drug1_ids = []
    drug2_ids = []
    cell_features = []
    labels = []
    for sample in batch:
        drug1_ids.append(sample[0])
        drug2_ids.append(sample[1])
        cell_features.append(torch.unsqueeze(sample[2], 0))
        labels.append(sample[3])

    cell_features = torch.concat(cell_features)
    labels = torch.tensor(labels)
    return drug1_ids, drug2_ids, cell_features, labels


def main(args=None):
    hparams = get_hparams(args)

    # CPU or GPU
    if torch.cuda.is_available():
        device_index = 'cuda:' + str(hparams['device'])
        device = torch.device(device_index)
        print('The code uses GPU...')
    else:
        device = torch.device('cpu')
        print('The code uses CPU!!!')

    data_df = pd.read_csv(hparams['data'])
    features_cell = pd.read_csv(hparams['f_cell'], index_col='Cell_Line_Name', dtype=str)
    graph_data = get_graph_data(features_drug_file=hparams['f_drug'],
                                features_target_file=hparams["f_target"],
                                e_dd_index_file=hparams["dd_edge"],
                                e_dt_index_file=hparams["dt_edge"],
                                e_tt_index_file=hparams["tt_edge"],
                                e_dd_att_file=hparams["dd_att"],
                                device=device)

    lenth = len(data_df)
    pot = int(lenth / 5)
    print('lenth', lenth)
    print('pot', pot)
    random_num = random.sample(range(0, lenth), lenth)

    modeling = UnnamedModel

    for i in range(5):

        online_model = modeling().to(device)
        target_model = modeling().to(device)

        initializes_target_network(online_model, target_model)

        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(online_model.parameters(), lr=hparams['lr'])
        # 学习率调整器，检测准确率的状态，然后衰减学习率
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, min_lr=1e-7, patience=5, verbose=True,
                                      threshold=0.0001, eps=1e-08)

        test_num = random_num[pot * i:pot * (i + 1)]
        train_num = random_num[:pot * i] + random_num[pot * (i + 1):]

        data_train = GNNDataset(label_df=data_df.iloc[train_num],
                                vocab_file=hparams['drug_vocab'],
                                features_cell_df=features_cell)
        data_test = GNNDataset(label_df=data_df.iloc[test_num],
                               vocab_file=hparams['drug_vocab'],
                               features_cell_df=features_cell)

        loader_train = DataLoader(data_train, batch_size=hparams['train_batch_size'], shuffle=None,
                                  collate_fn=batch_collate)
        loader_test = DataLoader(data_test, batch_size=hparams['test_batch_size'], shuffle=None,
                                 collate_fn=batch_collate)

        model_file_name = hparams['output'] + "GATNet(DrugA_DrugB)" + str(i) + '--model.model'
        file_AUCs = hparams['output'] + "GATNet(DrugA_DrugB)" + str(i) + '--AUCs.txt'
        AUCs = 'Epoch\tAUC_dev\tPR_AUC\tACC\tBACC\tPREC\tTPR\tKAPPA\tRECALL'
        with open(file_AUCs, 'w') as f:
            f.write(AUCs + '\n')

        best_auc = 0
        epochs = hparams['epoch']

        print('Training begin!')
        for epoch in range(epochs):
            # TODO
            train(device, graph_data, loader_train, loss_fn, online_model, optimizer, hparams['log_step'], epoch,
                  epochs, target_model=target_model, m=0.996)

            # T is correct label
            # S is predict score
            # Y is predict label
            T, S, Y = predict(online_model, device, loader_test, graph_data)

            print(T)
            print(S)
            print(Y)

            # compute preformence
            AUC = roc_auc_score(T, S)
            precision, recall, threshold = metrics.precision_recall_curve(T, S)
            PR_AUC = metrics.auc(recall, precision)
            BACC = balanced_accuracy_score(T, Y)
            tn, fp, fn, tp = confusion_matrix(T, Y).ravel()
            TPR = tp / (tp + fn)
            PREC = precision_score(T, Y)
            ACC = accuracy_score(T, Y)
            KAPPA = cohen_kappa_score(T, Y)
            recall = recall_score(T, Y)

            # save data
            if best_auc < AUC:
                best_auc = AUC
                AUCs = [epoch, AUC, PR_AUC, ACC, BACC, PREC, TPR, KAPPA, recall]

                with open(file_AUCs, 'a') as f:
                    f.write('\t'.join(map(str, AUCs)) + '\n')

                torch.save(online_model.state_dict(), model_file_name)

            print('best_auc', best_auc)

        scheduler.step(best_auc)


if __name__ == '__main__':
    main()
