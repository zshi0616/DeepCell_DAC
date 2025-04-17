NUM_PROC=2
GPUS=4,7
PORT=25668

python3 -m torch.distributed.launch --nproc_per_node=$NUM_PROC --master_port $PORT train_dg.py \
 --exp_id retrain_dg \
 --gpus $GPUS \
 --batch_size 32 --max_token_size 4096 
