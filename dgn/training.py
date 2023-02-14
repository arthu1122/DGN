import random

import pandas as pd
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.gat import HeteroGNN
from datasets.dataset import GNNDataset, get_graph_data

from sklearn.metrics import roc_curve, confusion_matrix
from sklearn.metrics import cohen_kappa_score, accuracy_score, roc_auc_score, precision_score, recall_score, \
    balanced_accuracy_score
from sklearn import metrics


# training function at each epoch
def train(model, device, loader_train, optimizer, epoch, graph_data):
    print('Training on {} samples...'.format(len(loader_train.dataset)))
    model.train()
    # train_loader = np.array(train_loader)
    for batch_idx, data in enumerate(loader_train):
        drug1_ids, drug2_ids, cell_features, labels = data

        cell_features = cell_features.to(device)
        labels = labels.to(device)
        # y = data[0].y.view(-1, 1).long().to(device)
        # y = y.squeeze(1)
        optimizer.zero_grad()
        output = model(drug1_ids, drug2_ids, cell_features, graph_data)
        loss = loss_fn(output, labels)
        # print('loss', loss)
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                           batch_idx * len(drug1_ids),
                                                                           len(loader_train.dataset),
                                                                           100. * batch_idx / len(loader_train),
                                                                           loss.item()))


def predicting(model, device, loader_test,graph_data):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    total_prelabels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader_test.dataset)))
    with torch.no_grad():
        for data in loader_test:
            drug1_ids, drug2_ids, cell_features, labels = data

            cell_features = cell_features.to(device)
            labels = labels.to(device)

            output = model(drug1_ids, drug2_ids, cell_features, graph_data)
            ys = F.softmax(output, 1).to('cpu').data.numpy()
            predicted_labels = list(map(lambda x: np.argmax(x), ys))
            predicted_scores = list(map(lambda x: x[1], ys))
            total_preds = torch.cat((total_preds, torch.Tensor(predicted_scores)), 0)
            total_prelabels = torch.cat((total_prelabels, torch.Tensor(predicted_labels)), 0)
            total_labels = torch.cat((total_labels, labels.cpu()), 0)
    return total_labels.numpy().flatten(), total_preds.numpy().flatten(), total_prelabels.numpy().flatten()


def batch_collate(batch):
    drug1_ids = []
    drug2_ids = []
    cell_features = []
    labels = []
    for sample in batch:
        drug1_ids.append(sample[0])
        drug2_ids.append(sample[1])
        cell_features.append(torch.unsqueeze(sample[2],0))
        labels.append(sample[3])

    cell_features = torch.concat(cell_features)
    labels = torch.tensor(labels)
    return drug1_ids, drug2_ids, cell_features, labels


modeling = HeteroGNN

TRAIN_BATCH_SIZE = 8
TEST_BATCH_SIZE = 8
LR = 0.0005
LOG_INTERVAL = 20
NUM_EPOCHS = 200

print('Learning rate: ', LR)
print('Epochs: ', NUM_EPOCHS)
datafile = 'new_labels_0_10'

# CPU or GPU
if torch.cuda.is_available():
    device = torch.device('cuda')
    print('The code uses GPU...')
else:
    device = torch.device('cpu')
    print('The code uses CPU!!!')

label_file = "../data/processed/Shuffled_Label_Data.csv"
features_drug_file = "../data/raw/Feature_DRUG.csv"
features_target_file = "../data/raw/Feature_TAR.csv"
features_cell_file = "../data/raw/Feature_CELL.csv"
e_dd_index_file = "../data/processed/drug_drug_edge_index.txt"
e_dt_index_file = "../data/processed/drug_tar_edge_index.txt"
e_tt_index_file = "../data/processed/tar_tar_edge_index.txt"
e_dd_att_file = "../data/processed/drug_drug_edge_att.txt"

label_df = pd.read_csv(label_file)
features_cell = pd.read_csv(features_cell_file, index_col='Cell_Line_Name', dtype=str)

graph_data = get_graph_data(features_drug_file=features_drug_file,
                            features_target_file=features_target_file,
                            e_dd_index_file=e_dd_index_file,
                            e_dt_index_file=e_dt_index_file,
                            e_tt_index_file=e_tt_index_file,
                            e_dd_att_file=e_dd_att_file)

lenth = len(label_df)
pot = int(lenth / 5)
print('lenth', lenth)
print('pot', pot)

random_num = random.sample(range(0, lenth), lenth)

for i in range(5):
    test_num = random_num[pot * i:pot * (i + 1)]
    train_num = random_num[:pot * i] + random_num[pot * (i + 1):]

    data_train = GNNDataset(label_df=label_df.iloc[train_num],
                            vocab_file="../data/processed/drug_vocab.txt",
                            features_cell_df=features_cell,
                            )
    data_test = GNNDataset(label_df=label_df.iloc[test_num],
                           vocab_file="../data/processed/drug_vocab.txt",
                           features_cell_df=features_cell,
                           )

    loader_train = DataLoader(data_train, batch_size=TRAIN_BATCH_SIZE, shuffle=None, collate_fn=batch_collate)
    loader_test = DataLoader(data_test, batch_size=TEST_BATCH_SIZE, shuffle=None, collate_fn=batch_collate)

    model = modeling().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    model_file_name = '../data/result/GATNet(DrugA_DrugB)' + str(i) + '--model_' + datafile + '.model'
    result_file_name = '../data/result/GATNet(DrugA_DrugB)' + str(i) + '--result_' + datafile + '.csv'
    file_AUCs = '../data/result/GATNet(DrugA_DrugB)' + str(i) + '--AUCs--' + datafile + '.txt'
    AUCs = ('Epoch\tAUC_dev\tPR_AUC\tACC\tBACC\tPREC\tTPR\tKAPPA\tRECALL')
    with open(file_AUCs, 'w') as f:
        f.write(AUCs + '\n')

    best_auc = 0
    for epoch in range(NUM_EPOCHS):
        train(model, device, loader_train, optimizer, epoch + 1,graph_data)
        T, S, Y = predicting(model, device, loader_test,graph_data)

        # T is correct label
        # S is predict score
        # Y is predict label

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

            # torch.save(model.state_dict(), model_file_name)
            # independent_num = []
            # independent_num.append(test_num)
            # independent_num.append(T)
            # independent_num.append(Y)
            # independent_num.append(S)
            # txtDF = pd.DataFrame(data=independent_num)
            # txtDF.to_csv(result_file_name, index=False, header=False)

        print('best_auc', best_auc)
