#!/bin/bash
#SBATCH -p mcml-dgx-a100-40x8
#SBATCH -q mcml
#SBATCH -n 8
#SBATCH --gres=gpu:8
#SBATCH --time=0-00:05:00
#SBATCH -o test.out
#SBATCH -e test.err
 
srun --container-mounts=/dss/dssmcmlfs01/pn57vo/pn57vo-dss-0000/Youssef/llm/:/media,/dss/dsshome1/0F/di35guq/:/workspace \
     --container-image='/dss/dsshome1/0F/di35guq/pytorch.sqsh' \
     torchrun --nproc-per-node 8 -m open_lm.main \
     --model open_lm_160m \
     --dataset-manifest /media/datasets/C4/manifest.jsonl \
     --train-num-samples 534000000 \
     --workers 8 \
     --precision amp_bfloat16 \
     --grad-checkpointing \
     --log-every-n-steps 100 \
     --grad-clip-norm 1 \
     --global-batch-size 32 \
     --data-key txt \
     --lr 3e-4 \
     --fsdp --fsdp-amp \
     --warmup 2000 \
     --wd 0.1 \
     --beta2 0.95 \
     --epochs 6 \
     --report-to wandb \
     --wandb-project-name test \
     --name test\
     --resume latest \
     --logs /media/logs/ \
     --classification True \
     --num-classes 50432 \
     --classif-model-path /media/logs/pretrain_C4_160m_3.2B/checkpoints/epoch_3.pt