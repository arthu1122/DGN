import argparse
import datetime
import glob
import os
import re

parser = argparse.ArgumentParser(
    description="calculate result.",
    usage="training.py [<args>] [-h | --help]"
)
parser.add_argument("--f", type=str, help="input")
parser.add_argument("--o", type=str, help="output")
parser.add_argument("--lr", type=float, help="learning rate")
parser.add_argument("--train_batch_size", type=int, help="train_batch_size")
parser.add_argument("--num_layers", type=int)
parser.add_argument("--hidden_channels", type=int)
parser.add_argument("--dropout", type=float)
parser.add_argument("--project1", type=int)
parser.add_argument("--project2", type=int)
parser.add_argument("--qk_dim", type=int)

args = parser.parse_args()

lr = args.lr
train_batch_size = args.train_batch_size
num_layers = args.num_layers
hidden_channels = args.hidden_channels
dropout = args.dropout
project1 = args.project1
project2 = args.project2
qk_dim = args.qk_dim


def get_config(args):
    config = "lr = {} " \
             "train_batch_size = {} " \
             "num_layers = {} " \
             "hidden_channels = {} " \
             "dropout = {} " \
             "project1 = {} " \
             "project2 = {} " \
             "qk_dim = {}".format(args.lr, args.train_batch_size, args.num_layers, args.hidden_channels, args.dropout, args.project1, args.project2, args.qk_dim, )
    return config


path = args.f + r"/*.txt"
files = []
for file in glob.glob(path):
    files.append(file)

print(files)


def get_last_line(inputfile):
    filesize = os.path.getsize(inputfile)
    blocksize = 1024
    dat_file = open(inputfile, 'rb')
    last_line = ""
    if filesize > blocksize:
        maxseekpoint = (filesize // blocksize)
        dat_file.seek((maxseekpoint - 1) * blocksize)
    elif filesize:
        # maxseekpoint = blocksize % filesize
        dat_file.seek(0, 0)
    lines = dat_file.readlines()
    if lines:
        last_line = lines[-1].strip()
    # print "last line : ", last_line
    dat_file.close()
    return last_line


AUC_dev = []
PR_AUC = []
ACC = []
BACC = []
PREC = []
TPR = []
KAPPA = []

for file in files:
    line = get_last_line(file)
    results=re.split("\\s+",line.decode())
    AUC_dev.append(float(results[1]))
    PR_AUC.append(float(results[2]))
    ACC.append(float(results[3]))
    BACC.append(float(results[4]))
    PREC.append(float(results[5]))
    TPR.append(float(results[6]))
    KAPPA.append(float(results[7]))

AUC_dev.append(sum(AUC_dev) / len(AUC_dev))
PR_AUC.append(sum(PR_AUC) / len(PR_AUC))
ACC.append(sum(ACC) / len(ACC))
BACC.append(sum(BACC) / len(BACC))
PREC.append(sum(PREC) / len(PREC))
TPR.append(sum(TPR) / len(TPR))
KAPPA.append(sum(KAPPA) / len(KAPPA))

import csv

with open(args.o, 'a', newline='') as f:
    writer = csv.writer(f)
    config = get_config(args)
    for i in range(6):
        if i != 5:
            writer.writerow([args.f,  i + 1, AUC_dev[i], PR_AUC[i], ACC[i], BACC[i], PREC[i], TPR[i], KAPPA[i],config])
        else:
            writer.writerow([args.f,  "AVG", AUC_dev[i], PR_AUC[i], ACC[i], BACC[i], PREC[i], TPR[i], KAPPA[i],config])
    writer.writerow([])
