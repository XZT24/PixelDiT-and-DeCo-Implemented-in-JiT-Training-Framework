# CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --nnodes=1 --node_rank=0 \
# main_jit.py \
# --model_type "DeCo" --model DeCo-XL/16 \
# --repa_weight 0.5 --repa_depth 8 --model_dim 1152 \
# --proj_dropout 0.0 \
# --P_mean -0.8 --P_std 0.8 \
# --img_size 256 --noise_scale 1.0 \
# --batch_size 32 --blr 1e-4 \
# --epochs 600 --warmup_epochs 0 \
# --gen_bsz 32 --num_images 50000 --cfg 3.0 --interval_min 0.1 --interval_max 0.9 \
# --output_dir "./train/" --resume "./train/" \
# --data_path "/home/zitong/Imagenet/" --online_eval \

CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --nnodes=1 --node_rank=0 \
main_jit.py \
--model_type "PixelDiT" --model PixelDiT-XL/16 \
--repa_weight 0.5 --repa_depth 8 --model_dim 1152 \
--proj_dropout 0.0 \
--P_mean -0.8 --P_std 0.8 \
--img_size 256 --noise_scale 1.0 \
--batch_size 32 --blr 1e-4 \
--epochs 600 --warmup_epochs 0 \
--gen_bsz 32 --num_images 50000 --cfg 3.0 --interval_min 0.1 --interval_max 0.9 \
--output_dir "./train/" --resume "./train/" \
--data_path "/home/zitong/Imagenet/" --online_eval \


