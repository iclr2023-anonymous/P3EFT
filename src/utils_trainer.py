import os

import evaluate
import numpy as np
import torch
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)

from utils_model import (
        compute_kmeans_acc, 
        compute_logreg_acc, 
        compute_spectral_auc, 
        compute_norm_auc
)

class TrainerWithMetrics(Trainer):
    # Custom class with compute_metrics function that has access to the Trainer object.

    def __init__(self, *args, task_name=None, is_regression=False, **kwargs):
        super().__init__(*args, **kwargs, compute_metrics=self.compute_metrics)
        self.is_regression = is_regression
        self.metric = self.get_metric(task_name, is_regression)

    def train(self, *args, **kwargs):
        # Run evaluation before training starts
        results = self.evaluate()

        # Now, you can either print the results or process them as needed
        print("results before evaluate: -----------------------")
        for key, value in results.items():
            print(f"{key}: {value}")
        print("----------------------------------------------")

        # Proceed with standard training
        super().train(*args, **kwargs)

    def evaluate(
        self,
        eval_dataset = None,
        ignore_keys = None,
        metric_key_prefix = "eval",
    ) :        
        if (self.state.global_step % 1000 == 0):  # edit to enable
            print ('SAVING CHECKPOINT')
            save_dir = self.args.output_dir[:-6] + "/checkpoints"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            save_dict = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
            save_path = os.path.join(save_dir, f"checkpoint_{self.state.global_step}.pt")
            print(f"Step {self.state.global_step}; Saving `checkpoint` to {save_path}")
            torch.save(save_dict, save_path)

        return super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
    
    @staticmethod
    def get_metric(task_name, is_regression):
        # Get the metric function
        if task_name is not None:
            metric = evaluate.load("glue", task_name)
        elif is_regression:
            metric = evaluate.load("mse")
        else:
            metric = evaluate.load("accuracy")
        return metric

    def compute_metrics(self, p: EvalPrediction):
        # original metrics from run_glue.py example
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if self.is_regression else np.argmax(preds, axis=1)
        result = self.metric.compute(predictions=preds, references=p.label_ids)
        if len(result) > 1:
            result["combined_score"] = np.mean(list(result.values())).item()

        result.update(self.compute_custom_metrics(p))
        return result

    def compute_custom_metrics(self, p: EvalPrediction):
        activations_kmeans = []
        activations_cv_logreg_score = []
        activations_spectral_auc = []
        activations_norm_auc = []
        for cur_lora_activations in p.predictions[1]:
            activations_kmeans.append(
                compute_kmeans_acc(cur_lora_activations, p.label_ids)
            )
            activations_cv_logreg_score.append(
                compute_logreg_acc(
                    cur_lora_activations, cur_lora_activations, p.label_ids, p.label_ids
                )[1]
            )
            activations_spectral_auc.append(
                compute_spectral_auc(cur_lora_activations, p.label_ids)
            )
            activations_norm_auc.append(
                compute_norm_auc(cur_lora_activations, p.label_ids)
            )

        shifts_kmeans = []
        shifts_cv_logreg_score = []
        shifts_spectral_auc = []
        shifts_norm_auc = []
        for cur_lora_shifts in p.predictions[2]:
            shifts_kmeans.append(compute_kmeans_acc(cur_lora_shifts, p.label_ids))
            shifts_cv_logreg_score.append(
                compute_logreg_acc(
                    cur_lora_shifts, cur_lora_shifts, p.label_ids, p.label_ids
                )[1]
            )
            shifts_spectral_auc.append(
                compute_spectral_auc(cur_lora_shifts, p.label_ids)
            )
            shifts_norm_auc.append(
                compute_norm_auc(cur_lora_shifts, p.label_ids)
            )

        custom_metrics_dict = {
            "activations_kmeans_max": np.max(activations_kmeans),
            "activations_kmeans_mean": np.mean(activations_kmeans),
            "activations_spectral_auc_max": np.max(activations_spectral_auc),
            "activations_spectral_auc_mean": np.mean(activations_spectral_auc),
            "activations_norm_auc_max": np.max(activations_norm_auc),
            "activations_norm_auc_mean": np.mean(activations_norm_auc),
            "activations_cv_logreg_max": np.max(activations_cv_logreg_score),
            "activations_cv_logreg_mean": np.mean(activations_cv_logreg_score),
            "shifts_kmeans_max": np.max(shifts_kmeans),
            "shifts_kmeans_mean": np.mean(shifts_kmeans),
            "shifts_spectral_auc_max": np.max(shifts_spectral_auc),
            "shifts_spectral_auc_mean": np.mean(shifts_spectral_auc),
            "shifts_norm_auc_max": np.max(shifts_norm_auc),
            "shifts_norm_auc_mean": np.mean(shifts_norm_auc),
            "shifts_cv_logreg_max": np.max(shifts_cv_logreg_score),
            "shifts_cv_logreg_mean": np.mean(shifts_cv_logreg_score),
        }
        return custom_metrics_dict
