NUM_PROC=4
GPUS=0,1,2,3
MASK=0.00
HOP=4

python3 -m torch.distributed.launch --nproc_per_node=$NUM_PROC train_mask.py \
 --exp_id mcm_${MASK}_${HOP} \
 --batch_size 4 --num_epochs 60 \
 --mask_ratio $MASK \
 --k_hop $HOP \
 --gpus $GPUS 