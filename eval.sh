CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 \
main_jit.py \
--model_type "PixelDiT" --model PixelDiT-XL/16 \
--img_size 256 --noise_scale 1.0 \
--gen_bsz 32 --num_images 50000 --cfg 3.0 --interval_min 0.1 --interval_max 0.9 \
--output_dir "./eval/" --resume "./train/" \
--data_path "/path/to/Imagenet/" --evaluate_gen \
--num_sampling_steps 50 --sampling_method "heun"