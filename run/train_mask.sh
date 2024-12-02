NUM_PROC=2
GPUS=6,7
MASK=0.01
PORT=29504
HOP=6

/opt/miniconda3/bin/python -m torch.distributed.launch --nproc_per_node=$NUM_PROC --master_port=$PORT train_mask.py \
 --exp_id 1118_mcm_${MASK}_${HOP} \
 --batch_size 8 --num_epochs 60 \
 --mask_ratio $MASK \
 --gpus $GPUS 