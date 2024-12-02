NUM_PROC=8
GPUS=0,1,2,3,4,5,6,7

/opt/miniconda3/bin/python -m torch.distributed.launch --nproc_per_node=$NUM_PROC --master_port=29956 train_eco.py \
 --exp_id eco \
 --num_epochs 60 \
 --gpus $GPUS 
