import logging
import os

import numpy as np
import peft
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from torch.utils.checkpoint import checkpoint
from transformers import (  # Trainer,; TrainingArguments,; default_data_collator,
    AutoConfig,
    T5ForSequenceClassification,
    AutoTokenizer,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

logger = logging.getLogger(__name__)


def get_antisymmetric_n01_coefs(seed=0, n_of_loras=2, neck_width=1536):
    set_seed(seed)
    antisymmetric_noise_matrix = torch.zeros((n_of_loras, n_of_loras, neck_width))
    for i in range(n_of_loras):
        for j in range(i + 1, n_of_loras):
            antisymmetric_noise_matrix[i, j] = torch.randn(neck_width)
            antisymmetric_noise_matrix[j, i] = -antisymmetric_noise_matrix[i, j]
    return antisymmetric_noise_matrix.sum(1)


def get_averaged_coefs(seed=0, n_of_loras=2, neck_width=1536):
    return torch.ones(n_of_loras) / n_of_loras


coefs_name_arr = ["antisymmetric_n01", "averaged"]
name2genfunc = {
    "antisymmetric_n01": get_antisymmetric_n01_coefs,
    "averaged": get_averaged_coefs,
}


def linear_combination_of_activations(activations_arr, coefs, add_noies=0):
    for i in range(len(activations_arr)):
        if i:
            activations += activations_arr[i] * coefs[i]
        else:
            activations = activations_arr[i] * coefs[i]
    return activations


def change_active_adapter(model, adapter_name):
    for m in model.modules():
        if isinstance(m, peft.tuners.lora.LoraLayer):
            m.active_adapter = adapter_name


def regularizing_logreg_loss(
    activations, labels, logreg, neck_width=1536, device="cuda:0"
):
    lin = nn.Linear(neck_width, 1, device=device)
    with torch.no_grad():
        lin.weight.copy_(torch.from_numpy(logreg.coef_))
        lin.bias.copy_(torch.from_numpy(logreg.intercept_))
    return F.binary_cross_entropy_with_logits(lin(activations).squeeze(), 1.0 - labels)


def compute_kmeans_acc(activations, true_labels):
    kmeans = KMeans(n_clusters=2, n_init=10).fit(activations)
    cluster_agreement = max(
        (kmeans.labels_ == true_labels).mean(), (kmeans.labels_ != true_labels).mean()
    )
    return cluster_agreement


def compute_logreg_acc(
    train_activations, val_activations, train_labels, val_labels, seed=0, cv=3
):
    logreg = get_fitted_logreg(train_activations, train_labels, seed=seed)
    val_acc = logreg.score(val_activations, val_labels)
    cv_acc = cross_val_score(logreg, train_activations, train_labels, cv=cv)
    return val_acc, np.mean(cv_acc)


def get_fitted_logreg(activations, labels, seed=0):
    logreg = LogisticRegression(random_state=seed, max_iter=150)
    logreg.fit(activations, labels)
    return logreg


class ModelWithMultipleLoras(torch.nn.Module):
    def __init__(
        self,
        base_model,
        get_head_input=linear_combination_of_activations,
        seed=0,
        device="cuda:0",
        n_of_loras=2,
        lora_rank=8,
        lora_alpha=8,
        lora_dropout=0.0,
        mult_std=1,
        method_name="averaged",
        activations_regularizing_weight=0.1,
        shift_regularizing_weight=0.1,
        loras_gradient_checkpointing=False,
        model_gradient_checkpointing=False,
    ):
        super().__init__()
        self.model = base_model
        
        self.model.classification_head = nn.Identity()
        if model_gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        self.loras_gradient_checkpointing = loras_gradient_checkpointing
        self.classifier = nn.Sequential()
#         self.classifier.add_module(f"dense", 
#                                    nn.Linear(self.deberta.config.d_model, self.deberta.config.d_model))
#         self.classifier.add_module(f"dropout",
#                                    nn.Dropout(p=0.0, inplace=False))
#         self.classifier.add_module(f"out_proj",
#                                    nn.Linear(self.deberta.config.d_model, 2))                   
        self.head_hidden_dims = [128, 32]
        self.head_hidden_dims.insert(0, self.model.config.d_model)
        for i in range(len(self.head_hidden_dims) - 1):
            self.classifier.add_module(
                f"linear{i}",
                nn.Linear(self.head_hidden_dims[i], self.head_hidden_dims[i + 1]),
            )
            self.classifier.add_module(f"relu{i}", nn.ReLU())
        self.classifier.add_module(
            "linear_final", nn.Linear(self.head_hidden_dims[-1], 2)
        )
        self.n_of_loras = n_of_loras
        self.get_head_input = get_head_input
        self.num_labels = 2
        self.mult_std = mult_std
        self.activations_regularizing_weight = activations_regularizing_weight
        self.shift_regularizing_weight = shift_regularizing_weight

        self.neck_width = self.model.config.d_model
        self.device = device
        self.seed = seed

        if method_name not in coefs_name_arr:
            raise NameError
        self.method_name = method_name
        self.coefs = (
            name2genfunc[self.method_name](
                seed=seed, n_of_loras=max(n_of_loras, 1), neck_width=self.neck_width
            )
            * mult_std
        ).to(device=device)
        if self.n_of_loras:
            peft_config = LoraConfig(
                peft_type=peft.PeftType.LORA,
                r=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
            )
            for i in range(self.n_of_loras):
                peft.get_peft_model(
                    self.model,
                    peft_config,
                    adapter_name=f"{self.method_name}_lora_{i}",
                )

    def _model_forward(self, input_ids, attention_mask, active_adapter):
        change_active_adapter(self.model, active_adapter)
        cur_lora_activations = self.model(
            input_ids, attention_mask=attention_mask
        ).logits
        return cur_lora_activations

    def forward(self, input_ids, attention_mask, labels):
        different_loras_activations = []
        for i in range(max(self.n_of_loras, 1)):
            if self.n_of_loras:
                active_adapter = f"{self.method_name}_lora_{i}"
            else:
                active_adapter = " "
            with torch.random.fork_rng(
                devices=(torch.device("cpu"), torch.device("cuda")), enabled=True
            ):
                if self.loras_gradient_checkpointing:
                    cur_lora_activations = checkpoint(
                        self._model_forward,
                        input_ids,
                        attention_mask,
                        active_adapter,
                        use_reentrant=False,
                    )
                else:
                    change_active_adapter(self.model, active_adapter)
                    cur_lora_activations = self.model(
                        input_ids,
                        attention_mask
                    ).logits
            different_loras_activations.append(cur_lora_activations)
        active_adapter = " "
        change_active_adapter(self.model, active_adapter)  # empty string = no adapter
        baseline_activations = self.model(
            input_ids, attention_mask=attention_mask
        ).logits

        activations = self.get_head_input(different_loras_activations, self.coefs)
        logits = self.classifier(activations)
        loss = F.cross_entropy(logits.view(-1, self.num_labels), labels.view(-1))
        
        if self.activations_regularizing_weight:
            for cur_activations in different_loras_activations:
                if len(torch.unique(labels)) < 2:
                    continue
                logreg = get_fitted_logreg(
                    cur_activations.detach().cpu().numpy(),
                    labels.cpu().numpy(),
                    seed=self.seed,
                )
                loss += (
                    regularizing_logreg_loss(
                        cur_activations,
                        labels,
                        logreg,
                        neck_width=self.neck_width,
                        device=self.device,
                    )
                    * self.activations_regularizing_weight
                )

        different_loras_shifts = []
        for cur_activations in different_loras_activations:
            cur_shift = cur_activations - baseline_activations
            different_loras_shifts.append(cur_shift)
        if self.shift_regularizing_weight:
            for cur_shift in different_loras_shifts:
                if len(torch.unique(labels)) < 2:
                    continue
                logreg = get_fitted_logreg(
                    cur_shift.detach().cpu().numpy(),
                    labels.cpu().numpy(),
                    seed=self.seed,
                )
                loss += (
                    regularizing_logreg_loss(
                        cur_shift,
                        labels,
                        logreg,
                        neck_width=self.neck_width,
                        device=self.device,
                    )
                    * self.shift_regularizing_weight
                )

        return (
            loss,
            logits,
            [cur_acts for cur_acts in different_loras_activations],
            different_loras_shifts,
            activations,
        )


def detect_last_checkpoint(training_args):
    # Detecting last checkpoint.
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif (
            last_checkpoint is not None and training_args.resume_from_checkpoint is None
        ):
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )


def get_base_model(model_args, finetuning_task, num_labels):
    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name
        if model_args.config_name
        else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=finetuning_task,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )
    #    print(model_args.use_fast_tokenizer, '\n\n\n')
    model = T5ForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
        ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
    )

    return model


def get_tokenizer(model_args):
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name
        if model_args.tokenizer_name
        else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )
    return tokenizer


def get_model_multiple_loras(base_model, model_args, training_args):
    print(
        "grad checkpointing. loras: ",
        model_args.loras_gradient_checkpointing,
        " model: ",
        model_args.model_gradient_checkpointing,
    )
    model_multiple_loras = ModelWithMultipleLoras(
        base_model=base_model,
        n_of_loras=model_args.n_of_loras,
        lora_rank=model_args.lora_rank,
        device=training_args.device,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
        seed=training_args.seed,
        mult_std=model_args.mult_std,
        method_name=model_args.coefs_method_name,
        activations_regularizing_weight=model_args.activations_regularizing_weight,
        shift_regularizing_weight=model_args.shift_regularizing_weight,
        loras_gradient_checkpointing=model_args.loras_gradient_checkpointing,
        model_gradient_checkpointing=model_args.model_gradient_checkpointing,
    )
    print(
        sum(
            p.numel()
            for p in model_multiple_loras.model.parameters()
            if p.requires_grad
        )
        / 1e6
    )
    print(
        sum(
            p.numel()
            for p in model_multiple_loras.classifier.parameters()
            if p.requires_grad
        )
        / 1e6
    )
    print(
        sum(p.numel() for p in model_multiple_loras.parameters() if p.requires_grad)
        / 1e6
    )

    print(
        model_args.lora_rank,
        model_args.lora_alpha,
        model_args.lora_dropout,
        model_args.n_of_loras,
        model_args.activations_regularizing_weight,
        model_args.shift_regularizing_weight,
    )
    print("device ", training_args.device)
    print("lr ", training_args.learning_rate)
    print("n epochs ", training_args.num_train_epochs)
    print("output dir ", training_args.output_dir)
    print("fp16 ", training_args.fp16)
    print("grad acum ", training_args.gradient_accumulation_steps)
    print("eval steps ", training_args.eval_steps)
    print("save steps ", training_args.save_steps)
    print("warmup steps ", training_args.warmup_steps)
    print("wd ", training_args.weight_decay)
    print("logging steps ", training_args.logging_steps)

    return model_multiple_loras
