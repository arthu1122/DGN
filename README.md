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
6. MAE 两个可学习的mask矩阵替代mask的节点 拉进两次输出的mask节点的表示 √
7. 找baseline 收集数据集


### 伪代码

```python

online_model = Model()
target_model = Model()

# 让 target参数 = online参数，grad=False
init_target(online_model, target_model)

# 优化器只作用于online的参数
optimizer = Adam(online_model.parameters())

for epoch in epoches:
    for data in dataloader:
        x_dict, edge_index_dict = graph_data

        # 输入的后面两个参数是定义在模型中的可学习的mask矩阵
        # 输出得到mask后的特征，drug的mask位置向量，target的mask位置向量
        _x_dict, drug_mask_index, target_mask_index = get_mask_x_dict(x_dict, mask_drug, mask_target)

        # online_model得到最后一层分类器的输出，和mask的图节点更新
        _output, _x_dict = online_model(drug1, durg2, cell, _x_dict, edge_index_dict)
        # target_model得到最后一层分类器的输出，和原图节点更新
        output, x_dict = target_model(drug1, drug2, cell, x_dict, edge_index_dict)

        # 用online_model的输出计算交叉熵得到第一个loss
        loss1 = cross_entropy(_output, labels)
        # 用两次得到的图结点，只计算被mask位置的结点的余弦误差（1-cos相似度）
        # 是只计算mask位置还是全部节点的？
        loss2 = cos_loss(x_dict, _x_dict, drug_mask_index, target_mask_index)

        # 用两个loss更新online_model
        loss = loss1 + loss2
        loss.backward()

        # 更新target_model 参数：
        # target_param = m * target_param + (1-m) * online_model
        update_target_parameters(online_model, target_model, m=0.996)

# 用online_model预测
predict(data, online_model)


```



