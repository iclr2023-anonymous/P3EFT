export CUDA_VISIBLE_DEVICES=0
export WANDB_DISABLED="true"
export CUBLAS_WORKSPACE_CONFIG=":16:8" # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
export PYTHONHASHSEED=0
export batch_size=64
export regularizing_weight=3.1
export lora_rank=8
export lora_alpha=8
export lora_dropout=0.0
export max_seq_length=128
export lr=6e-5
export lr_scheduler_type="linear"
export model_name="flan-t5-large"
export bf16=true
export n_epoch=16.0
export model_gradient_checkpointing=false
export seed=0
export output_dir="./sst2_baseline/$n_of_loras-loras"
python text-classification/run_glue_baseline_flan_t5.py \
--model_name_or_path google/$model_name \
--task_name sst2 \
--do_train \
--do_eval \
--max_seq_length $max_seq_length \
--per_device_train_batch_size $batch_size \
--per_device_eval_batch_size 64 \
--learning_rate $lr \
--num_train_epochs $n_epoch \
--output_dir $output_dir/model \
--bf16 $bf16 \
--gradient_accumulation_steps 1 \
--evaluation_strategy steps \
--eval_steps 200 \
--save_strategy no \
--warmup_steps 1000 \
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
