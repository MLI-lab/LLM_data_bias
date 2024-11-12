import torch
from open_lm.params import parse_args
import argparse
from open_lm.model import test_classif_model

args = parse_args([])
parser = argparse.ArgumentParser(description="Override params arguments with command-line arguments")
parser.add_argument('--model', type=str, help='Model name to use for evaluation')
parser.add_argument('--classif-model-path', type=str, help='Path to the classification model checkpoint')
cmd_args = parser.parse_args()
args.model = cmd_args.model
args.classif_model_path = cmd_args.classif_model_path



###########################################################################################################
args.num_classes = 6

path1 = "/media/datasets/RedPajama/val_seq/arxiv-shard-0000019.pt"
path2 = "/media/datasets/RedPajama/val_seq/c4-shard-0000019.pt"
path3 = "/media/datasets/RedPajama/val_seq/cc-shard-0000019.pt"
path4 = "/media/datasets/RedPajama/val_seq/gh-shard-0000019.pt"
path5 = "/media/datasets/RedPajama/val_seq/se-shard-0000019.pt"
path6 = "/media/datasets/RedPajama/val_seq/wiki-shard-0000009.pt"

str1 = "Arxiv"
str2 = "C4"
str3 =  "CC"
str4 = "Github"
str5 = "StackExchange"
str6 = "Wikipedia"
###########################################################################################################

model = test_classif_model(args)
model = model.to('cuda')



dataset = torch.load(path1)
sum = 0
for sample in dataset:
    sample = torch.LongTensor(sample).to('cuda')

    with torch.no_grad():
        out, _, _ = model(sample)
        
        pred = torch.argmax(out,2)[:,-1]
        
        n_correct = torch.sum(pred == 0).item()
        
        sum = sum + n_correct

sum1 = sum
len1 = len(dataset)
print(str1, sum1, "/" , len1)

##########################################################################################################################################################################################

dataset = torch.load(path2)
sum = 0
for sample in dataset:
    sample = torch.LongTensor(sample).to('cuda')
    
    with torch.no_grad():
        out, _, _ = model(sample)
        
        pred = torch.argmax(out,2)[:,-1]
        
        n_correct = torch.sum(pred == 1).item()
        
        sum = sum + n_correct

sum2 = sum
len2 = len(dataset)
print(str2, sum2, "/" , len2)

##########################################################################################################################################################################################

dataset = torch.load(path3)
sum = 0
for sample in dataset:
    sample = torch.LongTensor(sample).to('cuda')
    
    with torch.no_grad():
        out, _, _ = model(sample)
        
        pred = torch.argmax(out,2)[:,-1]
        
        n_correct = torch.sum(pred == 2).item()
        
        sum = sum + n_correct

sum3 = sum
len3 = len(dataset)
print(str3, sum3, "/" , len3)

##########################################################################################################################################################################################

dataset = torch.load(path4)
sum = 0
for sample in dataset:
    sample = torch.LongTensor(sample).to('cuda')
    
    with torch.no_grad():
        out, _, _ = model(sample)
        
        pred = torch.argmax(out,2)[:,-1]
        
        n_correct = torch.sum(pred == 3).item()
        
        sum = sum + n_correct

sum4 = sum
len4 = len(dataset)
print(str4, sum4, "/" , len4)

##########################################################################################################################################################################################

dataset = torch.load(path5)
sum = 0
for sample in dataset:
    sample = torch.LongTensor(sample).to('cuda')
    
    with torch.no_grad():
        out, _, _ = model(sample)
        
        pred = torch.argmax(out,2)[:,-1]
        
        n_correct = torch.sum(pred == 4).item()
        
        sum = sum + n_correct

sum5 = sum
len5 = len(dataset)
print(str5, sum5, "/" , len5)

##########################################################################################################################################################################################

dataset = torch.load(path6)
sum = 0
for sample in dataset:
    sample = torch.LongTensor(sample).to('cuda')
    
    with torch.no_grad():
        out, _, _ = model(sample)
        
        pred = torch.argmax(out,2)[:,-1]
        
        n_correct = torch.sum(pred == 5).item()
        
        sum = sum + n_correct

sum6 = sum
len6 = len(dataset)
print(str6, sum6, "/" , len6)

##########################################################################################################################################################################################



total_sum = sum1+sum2+sum3+sum4+sum5+sum6
total_length = len1+len2+len3+len4+len5+len6

print("Total= ", total_sum, "/" , total_length ) 
print("Accuracy= ", total_sum/total_length * 100, "%")
    

