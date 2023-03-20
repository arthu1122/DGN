# usage
# cd ~/data/DGN/dgn
# nohup sh bin/s1f3.sh > bin/sh_log/s1f3.log 2>&1 &

CUDA_VISIBLE_DEVICES=1,2,3, TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch ./training.py \
--setting=1 \
--train_batch_size=512 \
--test_batch_size=512 \
--lr=4e-5 \
--fold=3 \
--log_step=20 \
--num_workers=4 \
>bin/train_log/setting4fold1.log