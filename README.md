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
python -u training.py  > train.log

# or
# nohup sh  ./train.sh > ./sh.log 2>&1 &

```

## Note

1. 数据预处理
   * 标准化 -> tanh -> 标准化 
   * 去除方差为0的特征
2. 单机多卡
3. 找baseline 收集数据集
4. 加r-drop
5. 优化速度

gnn.convs.0.convs.drug__d-t__target.lin_dst.weight, 
gnn.convs.0.convs.drug__d-t__target.lin_src.weight, 
gnn.convs.0.convs.drug__d-t__target.bias, 
gnn.convs.0.convs.drug__d-t__target.att_dst, 
gnn.convs.0.convs.drug__d-t__target.att_src, 
gnn.convs.0.convs.target__t-t__target.att_src, 
gnn.convs.0.convs.target__t-t__target.att_dst, 
gnn.convs.0.convs.target__t-t__target.bias, 
gnn.convs.0.convs.target__t-t__target.lin_src.weight




