import os
import shutil
import random
import json
import torch
import numpy as np
import subprocess

from open_lm.params import parse_args
from open_lm.model import test_classif_model

device = "3"

def train_classifier(cuda_devices=device, log_dir="/workspace/youssef/lrz/logs/rewritten/classif160M_3.2BC4_C4_FW_320M_prompt3"):
    # Set the CUDA_VISIBLE_DEVICES environment variable
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices
    
    # Generate a random master port between 10000 and 65000
    master_port = random.randint(10000, 65000)

    # Construct the torchrun command
    command = [
        "torchrun",
        f"--master_port={master_port}",
        "--nproc-per-node", "1",
        "-m", "open_lm.main",
        "--model", "open_lm_160m",
        "--dataset-manifest", "/workspace/youssef/lrz/datasets/rewritten/0C4_1FW_prompt3/manifest.jsonl",
        "--train-num-samples", "320000000",
        "--workers", "1",
        "--precision", "amp_bfloat16",
        "--grad-checkpointing",
        "--log-every-n-steps", "100",
        "--grad-clip-norm", "1",
        "--global-batch-size", "16",
        "--data-key", "txt",
        "--lr", "3e-4",
        "--warmup", "2000",
        "--wd", "0.1",
        "--beta2", "0.95",
        "--epochs", "1",
        "--resume", "latest",
        "--logs", "/workspace/youssef/lrz/logs/rewritten/",
        "--name", "classif160M_3.2BC4_C4_FW_320M_prompt3",
        "--classification", "True",
        "--num-classes", "2",
        "--classif-model-path", "/workspace/youssef/lrz/logs/pretrain/160M_3.2BC4/checkpoint/epoch_3.pt"
    ]

    os.makedirs(log_dir, exist_ok=True)

    # Create log file paths
    stdout_log = os.path.join(log_dir, "output.log")
    stderr_log = os.path.join(log_dir, "error.log")

    # Run the torchrun command using subprocess
    with open(stdout_log, "w") as out_file, open(stderr_log, "w") as err_file:
        try:
            result = subprocess.run(command, check=True, stdout=out_file, stderr=err_file)
            print(f"torchrun finished with return code: {result.returncode}")
        except subprocess.CalledProcessError as e:
            print(f"An error occurred while running torchrun: {e}")


if __name__ == "__main__":
    print("starting script")
    train_classifier()
    print("ending script")
