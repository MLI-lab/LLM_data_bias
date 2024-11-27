This repository is based on the [Open LM](https://github.com/mlfoundations/open_lm) repository, which has been modified to allow for text classification. 

## Installation
We require python >=3.9 as well as several other packages. Start by cloning our project, and then installing the neccessary requirements as follows:

```
git clone https://github.com/MLI-lab/LLM_data_bias
cd LLM_data_bias
pip install -r requirements.txt
pip install --editable .
```

## Data Preparation

Check the [data preparation](https://github.com/MLI-lab/LLM_data_bias/tree/main/data_preparation) section for instructions on how to download and process the datasets.

## Pretraining

The classification model is first pretrained to predict the next token. Run the following command to run pretraining: 

```
torchrun --nproc-per-node 8 -m open_lm.main   \
 --model open_lm_160m \
 --dataset-manifest /preproc_data/manifest.jsonl \
 --train-num-samples 3200000000 \
 --epochs 1 \
 --workers 8 \
 --precision amp_bfloat16 \
 --global-batch-size 16 \
 --grad-checkpointing \
 --log-every-n-steps 100 \
 --grad-clip-norm 1 \
 --data-key txt \
 --lr 3e-4 \
 --fsdp --fsdp-amp \
 --warmup 2000 \
 --wd 0.1 \
 --beta2 0.95 \
 --resume latest \
 --report-to wandb \
 --wandb-project-name name_of_the_run \
 --logs path_to_logging_directory
 --name name_of_the_run \
```

Some of the important arguments are:

- `nproc-per-node`: Number of GPUs
- `model`: Model size, our default model size is 160M. The available model sizes can be found in [model_configs](https://github.com/MLI-lab/LLM_data_bias/tree/main/open_lm/model_configs)
- `dataset-manifest`: Path to the manifest file
- `train-num-samples`: Number of tokens per epoch. For the 160M model, 3.2B tokens are used (Chinchilla optimal)
- `epochs`: Model weights and optimizer are saved every epoch. To save intermediate checkpoints, set it to a higher value. For example setting `epochs` to 10, and `train-num-samples` to 320M will overall use 3.2B tokens
- `report-to wandb` and `wandb-project-name`: Omit if logging to wandb is not desired
- `logs`: Path where logging files and checkpoints are saved
- `name`: Project name. This creates a directory in `logs` with the project name


## Classification

The command for classification is similar to pretraining, but the following three arguments are added:

```
--classification True \
--num-classes 3 \
--classif-model-path path_to_pretrained_model
```

- `classification`: Indicates that we are doing classification not pretraining. Default value is False
- `num-classes`: Number of classification classes
- `classif-model-path`: Path to pretrained model. Can be omitted if you want to run classification from scratch, instead of finetuning from a pretrained model

