NUM_PROC=2
GPUS=4,7
MASK=0.00
HOP=4
PORT=25667

python3 -m torch.distributed.launch --nproc_per_node=$NUM_PROC --master_port $PORT train_mask.py \
 --exp_id linear_${MASK}_${HOP} \
 --batch_size 32 --num_epochs 60 --max_token_size 4096 \
 --mask_ratio $MASK \
 --k_hop $HOP \
 --linformer \
 --gpus $GPUS --resume 

