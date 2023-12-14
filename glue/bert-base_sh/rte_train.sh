#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export TASK_NAME=rte
export output_dir=./glue_out/tmp_base_rte/$TASK_NAME/

python run_glue.py \
  --model_name_or_path bert-base-cased \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --do_predict \
  --max_seq_length 128 \
  --per_device_train_batch_size 8 \
  --learning_rate 1e-5 \
  --num_train_epochs 8 \
  --save_steps 100 \
  --use_tlm 0 \
  --output_dir $output_dir

best_checkpoint=$(ls -d "$output_dir"/checkpoint-* | sort -V | tail -n 1)
echo "The best checkpoint directory is: $best_checkpoint"

python run_glue.py \
  --model_name_or_path $best_checkpoint \
  --task_name $TASK_NAME \
  --do_eval \
  --do_predict \
  --max_seq_length 128 \
  --output_dir $output_dir