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

### 20230216
1. 数据预处理
   * 标准化 -> tanh -> 标准化 
   * 去除方差为0的特征 
2. 训练文件参数化  √
3. 模型添加反向边的网络  √
4. 指标最后几个为0？ 
5. 单机多卡 
6. MAE 两个可学习的mask矩阵替代mask的节点 拉进两次输出的mask节点的表示
7. 找baseline 收集数据集



