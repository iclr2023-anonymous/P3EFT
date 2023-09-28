export CUDA_VISIBLE_DEVICES=0
export WANDB_DISABLED="true"
export CUBLAS_WORKSPACE_CONFIG=":16:8" # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
export PYTHONHASHSEED=0
export batch_size=256
export regularizing_weight=0.031
export lora_rank=8
export lora_alpha=8
export lora_dropout=0.0
export max_seq_length=128
export lr=2e-4
export lr_scheduler_type="linear"
export model_name="v2-xxlarge"
export fp16=true
export n_epoch=80
export model_gradient_checkpointing=true
export seed=0
export output_dir="./mrpc_baseline/$n_of_loras-loras"
python text-classification/run_glue_baseline.py \
--model_name_or_path microsoft/deberta-$model_name \
--task_name mrpc \
--do_train \
--do_eval \
--max_seq_length $max_seq_length \
--per_device_train_batch_size $batch_size \
--per_device_eval_batch_size 256 \
--learning_rate $lr \
--num_train_epochs $n_epoch \
--output_dir $output_dir/model \
--fp16 $fp16 \
--gradient_accumulation_steps 1 \
--evaluation_strategy steps \
--eval_steps 10 \
--save_strategy no \
--warmup_ratio 0.1 \
--seed $seed \
--weight_decay 0.01 \
--logging_steps 10 \
--logging_dir $output_dir/log \
--overwrite_output_dir \
--lora_rank $lora_rank \
--lora_alpha $lora_alpha \
--lora_dropout $lora_dropout \
--regularizing_weight $regularizing_weight \
--lr_scheduler_type $lr_scheduler_type \
--model_gradient_checkpointing $model_gradient_checkpointing \
