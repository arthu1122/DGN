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

torch.set_printoptions(threshold=np.inf)


def get_args(args):
    parser = argparse.ArgumentParser(
        description="Train a model.",
        usage="training.py [<args>] [-h | --help]"
    )

    parser.add_argument("--train_batch_size", type=int, default=128, help="train batch size")
    parser.add_argument("--test_batch_size", type=int, default=128, help="test batch size")
    parser.add_argument("--lr", type=float, default=1e-5, help="Path to pre-trained checkpoint.")
    parser.add_argument("--epochs", type=int, default=200, help="epoch for train")
    parser.add_argument("--device", type=str, default="0", help="device")
    parser.add_argument("--log_step", type=int, default=20, help="when to print log")
    parser.add_argument("--data", type=str, default="../data/processed/Shuffled_Label_Data.csv", help="training data with labels(.csv)")
    parser.add_argument("--f_drug", type=str, default="../data/raw/Feature_DRUG.csv", help="drug features file(.csv)")
    parser.add_argument("--f_target", type=str, default="../data/raw/Feature_TAR.csv", help="target features file(.csv)")
    parser.add_argument("--f_cell", type=str, default="../data/raw/Feature_CELL.csv", help="cell features file(.csv)")
    parser.add_argument("--dd_edge", type=str, default="../data/processed/drug_drug_edge_index.txt", help="drug-drug edge index file(.txt)")
    parser.add_argument("--dt_edge", type=str, default="../data/processed/drug_tar_edge_index.txt", help="drug-target edge index file(.txt)")
    parser.add_argument("--tt_edge", type=str, default="../data/processed/tar_tar_edge_index.txt", help="target-target edge index file(.txt)")
    parser.add_argument("--dd_att", type=str, default="../data/processed/drug_drug_edge_att.txt", help="drug-drug edge attribute file(.txt)")
    parser.add_argument("--drug_vocab", type=str, default="../data/processed/drug_vocab.txt", help="drug encoded file(.txt)")
    parser.add_argument("--output", type=str, default="../data/result/", help="output file fold")
    parser.add_argument("--target_model_update", type=float, default=0.996, help="leaving ratio of target_model")
    parser.add_argument("--mask_ratio", type=float, default=0.1, help="mae mask ratio")
    parser.add_argument("--kl", type=float, default=1.0, help="kl loss ratio")
    parser.add_argument("--mae", type=float, default=0.1, help="mae loss ratio")

    args = parser.parse_args(args)

    if not os.path.exists(args.output):
        os.mkdir(args.output)

    timestr = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    os.mkdir(args.output + timestr)
    args.output = args.output + timestr + "/"

    return args


def predict(model, device, loader_test, graph_data):
    model.to(device)
    model.eval()

    with torch.no_grad():
        total_preds = torch.Tensor()
        total_labels = torch.Tensor()
        total_predlabels = torch.Tensor()
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
            total_predlabels = torch.cat((total_predlabels, torch.Tensor(predicted_labels)), 0)
            total_preds = torch.cat((total_preds, torch.Tensor(predicted_scores)), 0)
            total_labels = torch.cat((total_labels, labels.cpu()), 0)

    return total_labels.numpy().flatten(), total_preds.numpy().flatten(), total_predlabels.numpy().flatten()


def train(device, graph_data, loader_train, loss_fn, online_model, optimizer, epoch, target_model, args):
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
        _x_dict, drug_mask_index, target_mask_index = get_mask_x_dict(x_dict, mask_drug, mask_target, ratio=args.mask_ratio)

        _output, _x_dict = online_model(drug1_ids, drug2_ids, cell_features, _x_dict, edge_index_dict)

        output1, x_dict1 = online_model(drug1_ids, drug2_ids, cell_features, x_dict, edge_index_dict)

        output, x_dict = target_model(drug1_ids, drug2_ids, cell_features, x_dict, edge_index_dict)

        loss0 = loss_fn(_output, labels)
        loss1 = loss_fn(output1, labels)

        loss_mae = get_mae_loss(x_dict, _x_dict, drug_mask_index, target_mask_index)

        loss_kl = nn.functional.kl_div(F.softmax(output1, 1).log(), F.softmax(_output, 1), reduction='batchmean')

        loss = loss0 + loss1 + args.kl * loss_kl + args.mae * loss_mae

        loss.backward()
        optimizer.step()

        update_target_network_parameters(online_model, target_model, args.target_model_update)

        if batch_idx % args.log_step == 0:
            print("[Train] {} Epoch[{}/{}] step[{}/{}] loss0={:.6f} loss1={:.6f} loss_kl={:.6f} loss_mae={:.6f} ".format(
                datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), epoch + 1, args.epochs, batch_idx + 1, len(loader_train), loss0.item(), loss1.item(), loss_kl.item(), loss_mae.item()))


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
    args = get_args(args)

    # CPU or GPU
    if torch.cuda.is_available():
        device_index = 'cuda:' + str(args.device)
        device = torch.device(device_index)
        print('The code uses GPU...')
    else:
        device = torch.device('cpu')
        print('The code uses CPU!!!')

    data_df = pd.read_csv(args.data)
    features_cell = pd.read_csv(args.f_cell, index_col='Cell_Line_Name', dtype=str)
    graph_data = get_graph_data(features_drug_file=args.f_drug, features_target_file=args.f_target, e_dd_index_file=args.dd_edge,
                                e_dt_index_file=args.dt_edge, e_tt_index_file=args.tt_edge, e_dd_att_file=args.dd_att, device=device)

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
        optimizer = torch.optim.Adam(online_model.parameters(), lr=args.lr)
        # 学习率调整器，检测准确率的状态，然后衰减学习率
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, min_lr=1e-7, patience=5, verbose=True, threshold=0.0001, eps=1e-08)

        test_num = random_num[pot * i:pot * (i + 1)]
        train_num = random_num[:pot * i] + random_num[pot * (i + 1):]

        data_train = GNNDataset(label_df=data_df.iloc[train_num], vocab_file=args.drug_vocab, features_cell_df=features_cell)
        data_test = GNNDataset(label_df=data_df.iloc[test_num], vocab_file=args.drug_vocab, features_cell_df=features_cell)

        loader_train = DataLoader(data_train, batch_size=args.train_batch_size, shuffle=None, collate_fn=batch_collate, num_workers=4, pin_memory=True)
        loader_test = DataLoader(data_test, batch_size=args.test_batch_size, shuffle=None, collate_fn=batch_collate, num_workers=4, pin_memory=True)

        model_file_name = args.output + str(i) + '--model.model'
        file_AUCs = args.output + str(i) + '--AUCs.txt'
        AUCs = "%-10s%-15s%-15s%-15s%-15s%-15s%-15s%-15s%-15s" % ('Epoch', 'AUC_dev', 'PR_AUC', 'ACC', 'BACC', 'PREC', 'TPR', 'KAPPA', 'RECALL')
        # AUCs = 'Epoch\tAUC_dev\tPR_AUC\tACC\tBACC\tPREC\tTPR\tKAPPA\tRECALL'
        with open(file_AUCs, 'w') as f:
            f.write(AUCs + '\n')

        best_auc = 0
        epochs = args.epochs

        print('Training begin!')
        for epoch in range(epochs):
            # TODO
            train(device, graph_data, loader_train, loss_fn, online_model, optimizer, epoch, target_model, args)

            # T is correct label
            # S is predict score
            # Y is predict label
            T, S, Y = predict(online_model, device, loader_test, graph_data)

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
                # AUCs = [epoch, AUC, PR_AUC, ACC, BACC, PREC, TPR, KAPPA, recall]
                AUCs = "%-10d%-15.8f%-15.8f%-15.8f%-15.8f%-15.8f%-15.8f%-15.8f%-15.8f" % (epoch, AUC, PR_AUC, ACC, BACC, PREC, TPR, KAPPA, recall)

                with open(file_AUCs, 'a') as f:
                    f.write(AUCs + '\n')

                torch.save(online_model.state_dict(), model_file_name)

            print('best_auc', best_auc)

        scheduler.step(best_auc)


if __name__ == '__main__':
    main()
