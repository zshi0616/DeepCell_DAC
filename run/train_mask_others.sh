NUM_PROC=4
GPUS=4,5,6,7
MASK=0.05
PORT=29503
HOP=4
GNN=gcn

/opt/miniconda3/bin/python -m torch.distributed.launch --nproc_per_node=$NUM_PROC --master_port=$PORT train_mask.py \
 --exp_id ${GNN}_${MASK}_${HOP} \
 --batch_size 8 --num_epochs 60 \
 --mask_ratio $MASK \
 --gnn $GNN \
 --gpus $GPUS 