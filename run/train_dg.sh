NUM_PROC=8
GPUS=0,1,2,3,4,5,6,7
/opt/miniconda3/bin/python -m torch.distributed.launch --nproc_per_node=$NUM_PROC train_dg.py \
 --exp_id dg \
 --gpus $GPUS \
 --batch_size 32
