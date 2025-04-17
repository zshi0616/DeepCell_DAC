NUM_PROC=4
GPUS=0,1,2,4
PORT=25666

python3 -m torch.distributed.launch --nproc_per_node=$NUM_PROC --master_port $PORT train_dc.py \
 --exp_id dc_train \
 --gpus $GPUS \
 --batch_size 64 --max_token_size 4096 
