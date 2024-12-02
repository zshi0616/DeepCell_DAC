NUM_PROC=4
GPUS=4,5,6,7
MODEL=gat
PORT=29505

/opt/miniconda3/bin/python -m torch.distributed.launch --nproc_per_node=$NUM_PROC --master_port=$PORT train_dc.py \
 --exp_id $MODEL \
 --gpus $GPUS \
 --batch_size 64 \
 --gnn $MODEL 
