#!/bin/bash
#SBATCH -p mcml-dgx-a100-40x8
#SBATCH -q mcml
#SBATCH -n 8
#SBATCH --gres=gpu:1
#SBATCH --time=0-00:05:00
#SBATCH -o /dss/dssmcmlfs01/pn57vo/pn57vo-dss-0000/Youssef/llm/logs/test/test.out
#SBATCH -e /dss/dssmcmlfs01/pn57vo/pn57vo-dss-0000/Youssef/llm/logs/test/test.err
 
srun --container-mounts=/dss/dssmcmlfs01/pn57vo/pn57vo-dss-0000/Youssef/llm/:/media,/dss/dsshome1/0F/di35guq/:/workspace \
     --container-image='/dss/dsshome1/0F/di35guq/pytorch.sqsh' \
     torchrun --nproc-per-node 1 -m open_lm.eval \
