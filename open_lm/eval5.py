import torch
from open_lm.params import parse_args
import argparse
from open_lm.model import test_classif_model

args = parse_args([])
parser = argparse.ArgumentParser(description="Override params arguments with command-line arguments")
parser.add_argument('--model', type=str, help='Model name to use for evaluation')
parser.add_argument('--classif-model-path', type=str, help='Path to the classification model checkpoint')
parser.add_argument('--str1', type=str, help='test set 1')
parser.add_argument('--str2', type=str, help='test set 2')
parser.add_argument('--str3', type=str, help='test set 3')
parser.add_argument('--str4', type=str, help='test set 4')
parser.add_argument('--str5', type=str, help='test set 5')
cmd_args = parser.parse_args()
args.model = cmd_args.model
args.classif_model_path = cmd_args.classif_model_path



###########################################################################################################

args.num_classes = 5

#Dolma_gen.pt
#DCLM_gen.pt
#FWEdu_gen.pt

#'C4.pt'
#'FineWeb.pt'
#'RefinedWeb.pt'


str1 = cmd_args.str1
str2 = cmd_args.str2
str3 = cmd_args.str3
str4 = cmd_args.str4
str5 = cmd_args.str5

base_path = '/media/datasets/test_set/'

data_path1 = base_path + str1 + '.pt'
data_path2 = base_path + str2 + '.pt'
data_path3 = base_path + str3 + '.pt'
data_path4 = base_path + str4 + '.pt'
data_path5 = base_path + str5 + '.pt'


###########################################################################################################

model = test_classif_model(args)
model = model.to('cuda')



dataset = torch.load(data_path1)
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

dataset = torch.load(data_path2)
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

dataset = torch.load(data_path3)
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

dataset = torch.load(data_path4)
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

dataset = torch.load(data_path5)
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

total_sum = sum1+sum2+sum3+sum4+sum5
total_length = len1+len2+len3+len4+len5

print("Total= ", total_sum, "/" , total_length ) 
print("Accuracy= ", total_sum/total_length * 100, "%")
    

