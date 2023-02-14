# DGN

## Step 1 Environment Creating:
```shell
conda create -n DGN python=3.8

conda activate DGN

# if cpu :
# pip install torch==1.12.1+cpu torchvision==0.13.1+cpu torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cpu
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113

# if cpu:
# pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.12.0+cpu.html
pip install pyg-lib torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-1.12.1+cu113.html
pip install torch-geometric

pip install pandas

```

## Step 2 Training:
```shell
python training.py -u > train.log

# or
# nohup sh  ./train.sh > ./sh.log 2>&1 &

```


