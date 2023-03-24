# usage
# cd ~/data/DGN/dgn
# nohup sh bin/all.sh > bin/sh_log/all.log 2>&1 &


visible_device1="1"
visible_device2="1"
visible_device3="2"
visible_device4="2"
# maybe 3
visible_device5="0"

# "gat" or "trmgat"
gnn="gat"
train_batch_size=512
lr=4e-5
setting=2
output="./bin/s2gat/"
num_workers=2

target_net_update=0.996
mask_ratio=0.1
kl=1.0
mae=0.1

num_layers=1
hidden_channels=768
dropout=0.2
project1=2048
project2=512
epochs=200
test_batch_size=512

# $1:fold
# $2:log output
# $3:port
# $4:visible devices
run() {
  nohup CUDA_VISIBLE_DEVICES="$4", TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port="$3" ./training.py \
    --gnn="${gnn}" \
    --setting="${setting}" \
    --output="${output}" \
    --train_batch_size="${train_batch_size}" \
    --test_batch_size="${test_batch_size}" \
    --lr="${lr}" \
    --fold="$1" \
    --log_step=1 \
    --num_workers=${num_workers} \
    --target_net_update=${target_net_update} \
    --mask_ratio=${mask_ratio} \
    --kl=${kl} \
    --mae=${mae} \
    --num_layers=${num_layers} \
    --hidden_channels=${hidden_channels} \
    --dropout=${dropout} \
    --project1=${project1} \
    --project2=${project2} \
    --epochs=${epochs} \
    >"$2"
}

run 1 "bin/train_log/f1.log" 29500 ${visible_device1}
run 2 "bin/train_log/f2.log" 29501 ${visible_device2}
run 3 "bin/train_log/f3.log" 29502 ${visible_device3}
run 4 "bin/train_log/f4.log" 29503 ${visible_device4}
run 5 "bin/train_log/f5.log" 29504 ${visible_device5}
