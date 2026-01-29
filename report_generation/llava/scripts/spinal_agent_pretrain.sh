#!/bin/bash

# IMPORTANT: this is the training script for the original LLaVA, NOT FOR LLaVA V1.5!

# Uncomment and set the following variables correspondingly to run this script:

# MODEL_VERSION=vicuna-v1-3-7b
MODEL_VERSION=llama3-8b-new

########### DO NOT CHANGE ###########
########### USE THIS FOR BOTH ###########
PROMPT_VERSION=spine_agent
########### DO NOT CHANGE ########### 
export NCCL_P2P_DISABLE=1

deepspeed --master_port 12540 --num_gpus 4 train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
    --version $PROMPT_VERSION \
    --data_path /path/to/pretrain_data.json \
    --img_data_path /path/to/image_data_full_paths.json \
    --text_data_path /path/to/clinical_notes.csv \
    --qa_data_dir /path/to/vqa_dataset \
    --agent_enabled True \
    --condition_classification_data_path /path/to/predicted_labels.json \
    --region_specific_images_data_path /path/to/structured_image_index.json \
    --top_1_similar_case_report_data_path /path/to/retrieval_top1.json \
    --exclude_patient_id_list /path/to/exclude_patient_ids.txt \
    --image_folder /path/to/mri_spinal_root/ \
    --vision_tower_t1 clip_align_t1 \
    --vision_tower_t2 clip_align_t2 \
    --model_path_t1 /path/to/dinov3_vitl_pretrain_t1/teacher_checkpoint.pth \
    --model_path_t2 /path/to/dinov3_vitl_pretrain_t2/teacher_checkpoint.pth \
    --config_file_t1 /path/to/dinov3_vitl_pretrain_t1/config.yaml \
    --config_file_t2 /path/to/dinov3_vitl_pretrain_t2/config.yaml \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -1 \
    --mm_projector_type "attn_pool+mlp2x_gelu" \
    --mm_hidden_size 1024 \
    --mm_context_size 1024 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --fp16 False \
    --output_dir ./spinal_checkpoints/agent/llava-$MODEL_VERSION-pretrain \
    --num_train_epochs 2 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 8096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb  \
    --pretrain_mode True \