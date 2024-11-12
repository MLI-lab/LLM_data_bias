import os
import shutil
import random
import json
import torch
import numpy as np
import subprocess

from open_lm.params import parse_args
from open_lm.model import test_classif_model

device = "cuda:3"

def inference():
    
    args = parse_args([])
    args.model = "open_lm_160m"
    args.classif_model_path = "/workspace/youssef/lrz/logs/rewritten/classif160M_3.2BC4_C4_FW_320M_prompt3/checkpoints/epoch_1.pt"
    args.num_classes = 2
    
    model = test_classif_model(args)
    model = model.to(device)


    test_data_path1 = '/workspace/youssef/lrz/datasets/test/rewritten/C4_test_prompt3.pt'
    test_data_path2 = '/workspace/youssef/lrz/datasets/test/rewritten/FW_test_prompt3.pt'
    
#####################################################################################################################    
    dataset = torch.load(test_data_path1)   
    sum = 0
    for sample in dataset:
        sample = torch.LongTensor(sample).to(device)
    
        with torch.no_grad():
            out, _, _ = model(sample)
            
            pred = torch.argmax(out,2)[:,-1]
            
            n_correct = torch.sum(pred == 0).item()
            
            sum = sum + n_correct
    
    sum1 = sum
    len1 = len(dataset)
    print('C4', sum1, "/" , len1)

    dataset = torch.load(test_data_path2)
    sum = 0
    for sample in dataset:
        sample = torch.LongTensor(sample).to(device)
        
        with torch.no_grad():
            out, _, _ = model(sample)
            
            pred = torch.argmax(out,2)[:,-1]
            
            n_correct = torch.sum(pred == 1).item()
            
            sum = sum + n_correct
    
    sum2 = sum
    len2 = len(dataset)
    print('FW', sum2, "/" , len2)
###############################################################################################################################

    total_sum = sum1+sum2
    total_length = len1+len2
    
    print("Total= ", total_sum, "/" , total_length ) 
    print("Accuracy= ", total_sum/total_length * 100, "%")


if __name__ == "__main__":
    print("starting script")
    inference()
    print("ending script")