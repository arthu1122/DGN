import argparse
import datetime
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from sklearn import metrics
from sklearn.metrics import cohen_kappa_score, accuracy_score, roc_auc_score, precision_score, balanced_accuracy_score
from sklearn.metrics import confusion_matrix
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from data.dataset import GNNDataset
from data.pipeline import GraphPipeline
from models.model import UnnamedModel
import random

import matplotlib.pyplot as plt

import matplotlib.colors as mcolors

torch.set_printoptions(threshold=np.inf)
accelerator = Accelerator()


def get_args(args):
    parser = argparse.ArgumentParser(
        description="Train a model.",
        usage="training.py [<args>] [-h | --help]"
    )
    # ------------- Train ------------------------
    parser.add_argument("--train_batch_size", type=int, default=128, help="train batch size")
    parser.add_argument("--test_batch_size", type=int, default=128, help="test batch size")
    parser.add_argument("--lr", type=float, default=1e-5, help="Path to pre-trained checkpoint.")
    parser.add_argument("--epochs", type=int, default=200, help="epoch for train")
    parser.add_argument("--device", type=str, default="0", help="device")
    parser.add_argument("--log_step", type=int, default=1, help="when to accelerator.print log")

    # ------------- Data ------------------------
    parser.add_argument("--data", type=str, default="../data/processed/fold_data/", help="training data with labels(.csv)")
    parser.add_argument("--fold", type=int, default=1, help="k-fold training index")
    parser.add_argument("--f_drug", type=str, default="../data/processed/feature/Feature_DRUG.csv", help="drug features file(.csv)")
    parser.add_argument("--f_target", type=str, default="../data/processed/feature/Feature_TAR.csv", help="target features file(.csv)")
    parser.add_argument("--f_cell", type=str, default="../data/processed/feature/Feature_CELL.csv", help="cell features file(.csv)")
    parser.add_argument("--dd_edge", type=str, default="../data/processed/edge_index/drug_drug_edge_index.txt", help="drug-drug edge index file(.txt)")
    parser.add_argument("--dt_edge", type=str, default="../data/processed/edge_index/drug_tar_edge_index.txt", help="drug-target edge index file(.txt)")
    parser.add_argument("--tt_edge", type=str, default="../data/processed/edge_index/tar_tar_edge_index.txt", help="target-target edge index file(.txt)")
    parser.add_argument("--dd_att", type=str, default="../data/processed/edge_att/drug_drug_edge_att.txt", help="drug-drug edge attribute file(.txt)")
    parser.add_argument("--drug_vocab", type=str, default="../data/processed/vocab/drug_vocab.txt", help="drug encoded file(.txt)")
    parser.add_argument("--output", type=str, default="./bin/result", help="output file fold")
    parser.add_argument("--num_workers", type=int, default=0, help="dataloader num_workers")

    parser.add_argument("--target_features_num", type=int, default=570, help="target features num")
    parser.add_argument("--drug_features_num", type=int, default=188, help="drug features num")
    parser.add_argument("--cell_features_num", type=int, default=890, help="cell features num")

    # ------------- Model ------------------------
    parser.add_argument("--target_net_update", type=float, default=0.996, help="leaving ratio of target_model")
    parser.add_argument("--mask_ratio", type=float, default=0.1, help="mae mask ratio")
    parser.add_argument("--kl", type=float, default=1.0, help="kl loss ratio")
    parser.add_argument("--mae", type=float, default=0.1, help="mae loss ratio")
    parser.add_argument("--setting", type=int, default=1, help="train type")
    parser.add_argument("--multi_model_settings", type=list, default=[2, 3], help="settings using target_model")

    parser.add_argument("--num_layers", type=int, default=1, help="gnn layer num")
    parser.add_argument("--hidden_channels", type=int, default=768, help="hidden channels in model")
    parser.add_argument("--qk_dim", type=int, default=768, help="qk dim")
    parser.add_argument("--dropout", type=float, default=0.2, help="dropout weight")
    parser.add_argument("--project1", type=int, default=2048, help="hidden channels in reduction 1st Linear")
    parser.add_argument("--project2", type=int, default=512, help="hidden channels in reduction 2nd Linear")
    parser.add_argument("--gnn", type=str, default='trmgat', help="type of gnn")

    args = parser.parse_args(args)

    if not os.path.exists(args.output):
        os.mkdir(args.output)

    args.model_output = args.output + "/" + str(args.fold) + '--model.model'
    args.result_output = args.output + "/" + str(args.fold) + '--AUCs.txt'

    return args


def predict(model, device, loader_test, graph_data):
    # model.to(device)
    model.eval()

    with torch.no_grad():
        total_preds = torch.Tensor()
        total_labels = torch.Tensor()
        total_predlabels = torch.Tensor()
        accelerator.print('Make prediction for {} samples...'.format(len(loader_test.dataset)))

        for step, data in enumerate(loader_test):
            accelerator.print("Dev Step[{}/{}]".format(step + 1, len(loader_test)))

            drug1_ids, drug2_ids, cell_features, labels = data

            x_dict = graph_data.node_dict
            mask = graph_data.mask
            edge_attr_dict = graph_data.edge_attr_dict
            output, _ = model(drug1_ids, drug2_ids, cell_features, x_dict, mask, edge_attr_dict)

            ys = F.softmax(output, 1).to('cpu').data.numpy()

            predicted_labels = list(map(lambda x: np.argmax(x), ys))
            predicted_scores = list(map(lambda x: x[1], ys))
            total_predlabels = torch.cat((total_predlabels, torch.Tensor(predicted_labels)), 0)
            total_preds = torch.cat((total_preds, torch.Tensor(predicted_scores)), 0)
            total_labels = torch.cat((total_labels, labels.cpu()), 0)

    return total_labels.numpy().flatten(), total_preds.numpy().flatten(), total_predlabels.numpy().flatten()


def train(device, graph_data, loader_train, loss_fn, online_model, optimizer, epoch, target_model, accelerator, args):
    online_model.train()
    loss_list = []
    for batch_idx, data in enumerate(loader_train):

        drug1_ids, drug2_ids, cell_features, labels = data

        optimizer.zero_grad()

        x_dict = graph_data.node_dict
        mask = graph_data.mask
        edge_attr_dict = graph_data.edge_attr_dict

        output, x_dict = online_model(drug1_ids, drug2_ids, cell_features, x_dict, mask, edge_attr_dict)
        loss = loss_fn(output, labels)
        loss_print = "loss={:.6f}".format(loss.item())
        loss_list.append(loss.item())
        accelerator.backward(loss)

        optimizer.step()

        for name, para in online_model.named_parameters():
            if para.grad is None:
                accelerator.print(name)

        if batch_idx % args.log_step == 0:
            accelerator.print("[Train] {} Epoch[{}/{}] step[{}/{}] ".format(
                datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), epoch + 1, args.epochs, batch_idx + 1, len(loader_train)) + loss_print)
    return loss_list


# get batch
def batch_collate(batch):
    drug1_ids = []
    drug2_ids = []
    cell_features = []
    labels = []

    for sample in batch:
        drug1_ids.append(sample[0])
        drug2_ids.append(sample[1])
        cell_features.append(sample[2].unsqueeze(0))
        labels.append(sample[3])

    drug1_ids = torch.concat(drug1_ids)
    drug2_ids = torch.concat(drug2_ids)
    cell_features = torch.concat(cell_features)
    labels = torch.concat(labels)

    return drug1_ids, drug2_ids, cell_features, labels


def main(args=None):
    args = get_args(args)

    if torch.cuda.is_available():
        device_index = 'cuda:' + args.device
        device = torch.device(device_index)
        accelerator.print('The code uses GPU...')
    else:
        device = torch.device('cpu')
        accelerator.print('The code uses CPU!!!')

    for k, v in sorted(vars(args).items()):
        accelerator.print(k, '=', v)

    # ----------- File Read ------------------------------------------------------
    features_drug = pd.read_csv(args.f_drug, index_col='DrugID', dtype=str)
    features_target = pd.read_csv(args.f_target, index_col='TargetID', dtype=str)
    with open(args.dd_edge, 'r') as f:
        edge_index_drug_drug = f.readlines()
    with open(args.dd_att, 'r') as f:
        edge_attr_drug_drug = f.readlines()
    with open(args.dt_edge, 'r') as f:
        edge_index_drug_target = f.readlines()
    with open(args.tt_edge, 'r') as f:
        edge_index_target_target = f.readlines()

    train_path = args.data + "fold_" + str(args.fold) + "_train.csv"
    test_path = args.data + "fold_" + str(args.fold) + "_test.csv"
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    features_cell = pd.read_csv(args.f_cell, index_col='Cell_Line_Name', dtype=str)

    with open(args.drug_vocab, 'r') as f:
        vocab = f.readlines()
        _ID2id = {}
        for i in range(len(vocab)):
            _ID2id[vocab[i][:-1]] = i

    # ----------- Hparam Setting ---------------------------------------------------

    train_batch_size_list = [128, 512]

    num_layers_list = [1, 2, 3]
    hidden_channels_list = [768, 1536, 2304]
    dropout_list = [0.1, 0.2, 0.3]
    project1_list = [512, 1024, 2048]
    project2_list = [512, 1024, 2048]
    qk_dim_list = [384, 768,1536]

    search_step = 10
    epochs = 50




    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(20, 6), dpi=100)

    colors = list(mcolors.TABLEAU_COLORS.keys())
    for step in range(search_step):
        args.lr = 10 ** np.random.uniform(-6, -2)
        args.train_batch_size = random.choice(train_batch_size_list)
        args.num_layers = random.choice(num_layers_list)
        args.hidden_channels = random.choice(hidden_channels_list)
        args.dropout = random.choice(dropout_list)
        args.project1 = random.choice(project1_list)
        args.project2 = random.choice(project2_list)
        args.qk_dim = random.choice(qk_dim_list)
        args.epochs = epochs

        with open("setting.txt", 'a') as f:
            f.write("setting "+str(step+1))
            f.write("lr = "+str(args.lr)+"\n")
            f.write("batch = " + str(args.train_batch_size) + "\n")
            f.write("num_layers = " + str(args.num_layers) + "\n")
            f.write("hidden = " + str(args.hidden_channels) + "\n")
            f.write("dropout = " + str(args.dropout) + "\n")
            f.write("p1 = " + str(args.project1) + "\n")
            f.write("p2 = " + str(args.project2) + "\n")
            f.write("qk_dim = " + str(args.qk_dim) + "\n")
            f.write("\n")

        train_data = train_data.sample(frac=0.5, replace=False)
        test_data = test_data.sample(frac=0.5, replace=False)

        loss_list, auc_list = search(_ID2id, args, device, edge_attr_drug_drug, edge_index_drug_drug, edge_index_drug_target, edge_index_target_target, features_cell, features_drug, features_target,
                                     test_data, train_data)

        axs[0].plot(loss_list, c=mcolors.TABLEAU_COLORS[colors[step]], label="setting" + str(step + 1))
        axs[1].plot(auc_list, c=mcolors.TABLEAU_COLORS[colors[step]], label="setting" + str(step + 1))
    fig.autofmt_xdate()
    plt.show()
    plt.savefig("./result.png")

def search(_ID2id, args, device, edge_attr_drug_drug, edge_index_drug_drug, edge_index_drug_target, edge_index_target_target, features_cell, features_drug, features_target, test_data, train_data):
    # ----------- Data Prepare ---------------------------------------------------
    graph_data = GraphPipeline.get_data(features_drug, features_target, edge_index_drug_drug,
                                        edge_index_drug_target, edge_index_target_target, edge_attr_drug_drug, accelerator.device)
    data_train = GNNDataset(train_data, _ID2id, features_cell, accelerator.device)
    data_test = GNNDataset(test_data, _ID2id, features_cell, accelerator.device)
    loader_train = DataLoader(data_train, batch_size=args.train_batch_size, shuffle=True, collate_fn=batch_collate,
                              num_workers=args.num_workers, pin_memory=True)
    loader_test = DataLoader(data_test, batch_size=args.test_batch_size, shuffle=False, collate_fn=batch_collate,
                             num_workers=args.num_workers, pin_memory=True)
    # ----------- Model Prepare ---------------------------------------------------
    modeling = UnnamedModel
    # online_model = modeling(args).to(device)
    online_model = modeling(args)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(online_model.parameters(), lr=args.lr)
    # 学习率调整器，检测准确率的状态，然后衰减学习率
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, min_lr=1e-7, patience=5, verbose=True, threshold=0.0001, eps=1e-08)
    accelerator.print("model:", online_model)
    # ----------- Output Prepare ---------------------------------------------------
    AUCs = "%-10s%-15s%-15s%-15s%-15s%-15s%-15s%-15s%-15s" % ('Epoch', 'AUC_dev', 'PR_AUC', 'ACC', 'BACC', 'PREC', 'TPR', 'KAPPA', 'RECALL')
    # AUCs = 'Epoch\tAUC_dev\tPR_AUC\tACC\tBACC\tPREC\tTPR\tKAPPA\tRECALL'
    with open(args.result_output, 'w') as f:
        for k, v in sorted(vars(args).items()):
            f.write(str(k) + '=' + str(v) + "\n")
        f.write(str(online_model) + '\n')
        f.write(AUCs + '\n')
    # ----------- Training ---------------------------------------------------
    best_auc = 0
    online_model, optimizer, loader_train, loader_test, scheduler = accelerator.prepare(online_model, optimizer, loader_train, loader_test, scheduler)
    epochs = args.epochs
    accelerator.print('Training begin!')
    loss_list = []
    auc_list = []
    for epoch in range(epochs):
        loss_list = train(device, graph_data, loader_train, loss_fn, online_model, optimizer, epoch, None, accelerator, args)

        loss = sum(loss_list) / len(loss_list)

        T, S, Y = predict(online_model, device, loader_test, graph_data)

        # compute performance
        AUC = roc_auc_score(T, S)
        precision, recall, threshold = metrics.precision_recall_curve(T, S)
        PR_AUC = metrics.auc(recall, precision)
        BACC = balanced_accuracy_score(T, Y)
        tn, fp, fn, tp = confusion_matrix(T, Y).ravel()
        TPR = tp / (tp + fn)
        PREC = precision_score(T, Y)
        ACC = accuracy_score(T, Y)
        KAPPA = cohen_kappa_score(T, Y)

        # save data
        if best_auc < AUC:
            best_auc = AUC
        # AUCs = [epoch, AUC, PR_AUC, ACC, BACC, PREC, TPR, KAPPA, recall]
        AUCs = "%-10d%-15.8f%-15.8f%-15.8f%-15.8f%-15.8f%-15.8f%-15.8f" % (epoch, AUC, PR_AUC, ACC, BACC, PREC, TPR, KAPPA)

        with open(args.result_output, 'a') as f:
            f.write(AUCs + '\n')

        accelerator.print('best_auc', best_auc)
        scheduler.step(best_auc)

        loss_list.append(loss)
        auc_list.append(AUC)

    return loss_list, auc_list


if __name__ == '__main__':
    main()
