import os
import shutil
import random
import json
import torch
import numpy as np
import subprocess

from open_lm.params import parse_args
from open_lm.model import test_classif_model

def inference():
    
    args = parse_args([])
    args.model = "open_lm_25m"
    args.classif_model_path = "/workspace/youssef/lrz/logs/RedPajama/prop/checkpoints/epoch_1.pt"
    args.num_classes = 2
     
    test_data_path = '/workspace/youssef/lrz/datasets/prop/Llama1_gen.pt'
    dataset = torch.load(test_data_path)
    
    model = test_classif_model(args)
    model = model.to('cuda:3')
       
    pred = []
    for sample in dataset:
        sample = torch.LongTensor(sample).to('cuda:3')
        with torch.no_grad():
            out, _, _ = model(sample)        
            pred.append(torch.argmax(out,2)[:,-1].item())
               
    c1 = pred.count(0)
    c2 = pred.count(1)

    print(c1,c2)

    if c2 > c1:
        return 1
    else:
        return 0

def train_classifier(cuda_devices="3", log_dir="/workspace/youssef/lrz/logs/RedPajama/prop"):
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
        "--model", "open_lm_25m",
        "--dataset-manifest", "/workspace/youssef/lrz/datasets/prop/train/manifest.jsonl",
        "--train-num-samples", "200000000",
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
        "--logs", "/workspace/youssef/lrz/logs/RedPajama/",
        "--name", "prop",
        "--classification", "True",
        "--num-classes", "2",
        "--classif-model-path", "/workspace/youssef/lrz/logs/pretrain/25M_0.5BC4/checkpoint/epoch_1.pt"
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



def proj_simplex(y):
    m = len(y)
    bget = False
    s = sorted(y, reverse=True) # sorting in descending order
    tmpsum = 0
    for i in range(m-1):
        tmpsum = tmpsum + s[i]
        tmax = (tmpsum - 1) / (i+1)
        if tmax >= s[i+1]:
            bget = True
            break
    if not bget:
        tmax = (tmpsum + s[m-1] -1) / m
    return np.maximum(y-tmax,0)



def del_dir(dir_path):
    try:
        # Remove the directory and all its contents
        shutil.rmtree(dir_path)
        print(f"Removed directory: {dir_path}")
    except FileNotFoundError:
        print(f"Directory not found: {dir_path}")
    except PermissionError:
        print(f"Permission denied: {dir_path}")
    except Exception as e:
        print(f"An error occurred while removing the directory: {e}")


def round_preserving_sum(numbers):
    """
    This function takes a list of numbers that add up to 1, multiplies each by 100,
    rounds them to integers while preserving the sum as 100.
    """
    # Step 1: Multiply all numbers by 100
    multiplied = np.array(numbers) * 100

    # Step 2: Separate integer and decimal parts
    integers = np.floor(multiplied).astype(int)  # Integer parts
    decimals = multiplied - integers  # Decimal parts

    # Step 3: Calculate the difference between the current sum and 100
    current_sum = np.sum(integers)
    difference = 100 - current_sum

    # Step 4: Distribute the difference by rounding up the largest decimals
    if difference > 0:
        # Get indices of the largest decimals and round up those numbers
        indices_to_round_up = np.argsort(-decimals)[:difference]
        integers[indices_to_round_up] += 1

    return integers.tolist()

def sample_and_rename_files(sample_counts_list):

    base_path = "/workspace/youssef/lrz/datasets/prop/original/"
    output_folder = "/workspace/youssef/lrz/datasets/prop/train/"
   
    # Define the folder names in order
    file_names = ['arxiv', 'c4', 'cc', 'github', 'se', 'wiki']
    folder_names = [os.path.join(base_path, folder) for folder in file_names]
    
    # Check if the provided sample_counts_list contains exactly two lists
    if len(sample_counts_list) != 2 or any(len(sample_counts) != 6 for sample_counts in sample_counts_list):
        raise ValueError("sample_counts_list must contain exactly two lists, each with 6 numbers.")
    
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # List to store the manifest data
    manifest_data = []

    # Loop over the two lists of sample counts
    for index, sample_counts in enumerate(sample_counts_list):
        # Iterate through each folder and sample the required number of .tar files
        for i, folder in enumerate(folder_names):
            folder_path = os.path.join(folder)
            
            if not os.path.exists(folder_path):
                raise ValueError(f"Folder {folder_path} does not exist.")
            
            # Get all .tar files from the current folder
            all_files = [f for f in os.listdir(folder_path) if f.endswith('.tar')]
            
            # Ensure the sample count is not more than available files
            sample_count = min(sample_counts[i], len(all_files))
            
            # Randomly sample the required number of files from the folder
            sampled_files = random.sample(all_files, sample_count)
            
            # Copy each sampled file to the output folder with the new name
            for file_name in sampled_files:
                # Construct source file path
                source_file_path = os.path.join(folder_path, file_name)
                
                # Create the new filename by prepending the index (0 or 1) with a dash
                new_file_name = f"{index}-{file_name[:-4]}"  # Remove the .tar extension
                
                # Destination path in the output folder
                dest_file_path = os.path.join(output_folder, new_file_name + '.tar')  # Keep .tar in destination
                
                # Copy the file to the output folder
                shutil.copy2(source_file_path, dest_file_path)
                
                # Add entry to manifest_data, replacing ".tar" in new_file_name with an empty string
                manifest_entry = {
                    "shard": new_file_name,  # No .tar extension
                    "num_sequences": 489  # Set a fixed number of sequences
                }
                manifest_data.append(manifest_entry)

    # Write the manifest.jsonl file
    manifest_file_path = os.path.join(output_folder, "manifest.jsonl")
    with open(manifest_file_path, 'w') as manifest_file:
        # Write each entry except the last one with a newline
        for entry in manifest_data:
            manifest_file.write(json.dumps(entry) + '\n')

    print(f"Files sampled and saved in {output_folder}. Manifest file created as {manifest_file_path}.")