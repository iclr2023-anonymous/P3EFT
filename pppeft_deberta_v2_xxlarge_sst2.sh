export CUDA_VISIBLE_DEVICES=0
export WANDB_DISABLED="true"
export CUBLAS_WORKSPACE_CONFIG=":16:8" # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
export PYTHONHASHSEED=0
export n_of_loras=2
export batch_size=32
export coefs_method_name="antisymmetric_n01"
export activations_regularizing_weight=0.31
export shift_regularizing_weight=0.31
export mult_std=31
export lora_rank=8
export lora_alpha=8
export lora_dropout=0.0
export max_seq_length=128
export lr=6e-5
export lr_scheduler_type="linear"
export model_name="v2-xxlarge"
export fp16=true
export n_epoch=16.0
export seed=1
export loras_gradient_checkpointing=false
export model_gradient_checkpointing=false
export output_dir="./sst2/$n_of_loras-loras"
python src/main.py \
--model_name_or_path microsoft/deberta-$model_name \
--task_name sst2 \
--do_train \
--do_eval \
--max_seq_length $max_seq_length \
--per_device_train_batch_size $batch_size \
--per_device_eval_batch_size 64 \
--learning_rate $lr \
--num_train_epochs $n_epoch \
--output_dir $output_dir/model \
--fp16 $fp16 \
--gradient_accumulation_steps 1 \
--evaluation_strategy steps \
--eval_steps 200 \
--save_strategy no \
--warmup_steps 2000 \
--seed $seed \
--weight_decay 0.01 \
--logging_steps 10 \
--logging_dir $output_dir/log \
--overwrite_output_dir \
--n_of_loras $n_of_loras \
--lora_rank $lora_rank \
--lora_alpha $lora_alpha \
--lora_dropout $lora_dropout \
--mult_std $mult_std \
--coefs_method_name $coefs_method_name \
--activations_regularizing_weight $activations_regularizing_weight \
--shift_regularizing_weight $shift_regularizing_weight \
--lr_scheduler_type $lr_scheduler_type \
--loras_gradient_checkpointing $loras_gradient_checkpointing \
--model_gradient_checkpointing $model_gradient_checkpointing \
