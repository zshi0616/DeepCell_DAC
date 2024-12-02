NUM_PROC=4
GPUS=2,3,5,7
python3 -m torch.distributed.launch --nproc_per_node=$NUM_PROC train_dc.py \
 --gpus $GPUS \
 --batch_size 4
