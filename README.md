# Supplementary code for Privacy Preserving API Fine-tuning for LLMs

## How to install

First, install the requirements: `pip install -r requirements.txt`

Then clone transformers for the specified revision (v4.33.x)
```bash
git clone https://github.com/huggingface/transformers
cd transformers
git checkout -b bffac926ca6bc6c965a92bfbfd00c567a2c0fb90
```

Finally, add the entire contents of this repository to transformers/examples/pytorch/

```
git clone https://github.com/iclr2023-anonymous/P3EFT
cp -r P3EFT/* $TRANSFORMERS_PATH_HERE/examples/pytorch/
```

## How to run

The main folder contains 6 bash scripts that run main fine-tuning experiments from section 4.2. These scripts have the following names: {method_name} {model_name} {task_name}.

There are two possible choices for models: [DeBERTa xxlarge (He et al., 2020)](https://arxiv.org/abs/2006.03654) (method_name == deberta_v2_xxlarge) and [Flan-T5 (Chung et al., 2022)](https://arxiv.org/abs/2210.11416) (method_name == flan-t5_large), as well as for the datasets: SST2 (task_name == sst2) and MRPC (task_name == mrpc) from [GLUE benchmark (Wang et al., 2018)](https://arxiv.org/abs/1804.07461).

We present experiments for our method P^3EFT (method_name == pppeft) and for our implementation of the [Distance Correlation method (Sun et al., 2022)](https://arxiv.org/abs/2203.01451) (method_name == baseline). In both cases, there are some number of important arguments.

* Common arguments for `transformers.Trainer` (such that `batch_size`, `max_seq_length`, `lr`, `lr_scheduler_type`, `n_epoch` etc.) and special arguments for [LoRA (Hu et al., 2022)](https://arxiv.org/abs/2106.09685) (`lora_rank`, `lora_alpha`, `lora_dropout`). In most cases, we took these arguments from the [original paper](https://arxiv.org/abs/2106.09685), except for the `n_epoch` for MRPC and `batch_size` for SST2.

* Hyperparameters for privacy protection methods. 

* `P^3EFT` has 5 hyperparameters: `n_of_loras`, `mult_std`, `activations_regularizing_weight`, `shift_regularizing_weight` and `coefs_method_name`. 

    * `n_of_loras` is responsible for the number of different copies of adapters that are individually inserted into the model. 

    * After receiving the activations for each individual copy of adapters we mix them with specific coefficients. The magnitude of the norm of these coefficients can be changed via `mult_std`. More specifically, the random vectors used hereafter to obtain the coefficients are initially generated from a normal distribution with variance `1` and then multiplied by `mult_std``.
    
    * `activations_regularizing_weight` and `shift_regularizing_weight` adjust the weights of the corresponding regularization loss functions. Basically we took them equal for our experiments, but you can try to vary them separately.

    * `coefs_method_name` is an auxiliary parameter which can take 2 values: 'antisymmetric_n01' and 'averaged'. The first value is the main one for experiments with our method, and the second one is needed only to run the baseline without regularization (together with the values `n_of_loras = 1`, `mult_std = 1`, `activations_regularizing_weight = shift_regularizing_weight = 0`).

[Distance Correlation method (Sun et al., 2022)](https://arxiv.org/abs/2203.01451) has 1 hyperparameter: `regularizing_weight` which has the similar meaning as `activations_regularizing_weight` and `shift_regularizing_weight` in `P^3EFT`.

The scripts have default values, which were used for the main figures in the article.

* In order to reduce the amount of GPU RAM you can use 3 additional flags:

    * Above all, we advise to turn on `fp16` for DeBERTa or `bf16` for Flan-T5.

    * `loras_gradient_checkpointing` creates checkpoints between adapter set switches. Thus the amount of memory consumed will be the same as in the case of a single adapter set.

    * `model_gradient_checkpointing` turns on regular `transformers.Trainer` chechpoints. Cannot be used without `loras_gradient_checkpointing`, since the regular checkpointing system does not switch adapter sets, which is essential to run our method.

    * We don't advise to use `gradient_accumulation` and `torch.nn.DataParallel` techniques, since then the loss is computed separately on each minibatch, while both privacy protection methods assume the computation of the regularization loss over the entire batch.

## Additional experiments

Notebooks `vis_1_get_acts_grads.ipynb` and `vis_2_draw_charts.ipynb` can be used to reproduce Figure 3.