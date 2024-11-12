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
cmd_args = parser.parse_args()
args.model = cmd_args.model
args.classif_model_path = cmd_args.classif_model_path



###########################################################################################################

args.num_classes = 3

#Dolma_gen.pt
#DCLM_gen.pt
#FWEdu_gen.pt

#'C4.pt'
#'FineWeb.pt'
#'RefinedWeb.pt'


str1 = cmd_args.str1
str2 = cmd_args.str2
str3 = cmd_args.str3

base_path = '/media/datasets/test_set/'

data_path1 = base_path + str1 + '.pt'
data_path2 = base_path + str2 + '.pt'
data_path3 = base_path + str3 + '.pt'


###########################################################################################################

model = test_classif_model(args)
model = model.to('cuda')


indices = torch.arange(0, 2048, 200)

sum = torch.zeros(len(indices))

dataset = torch.load(data_path1)

for sample in dataset:
    sample = torch.LongTensor(sample).to('cuda')

    with torch.no_grad():
        out, _, _ = model(sample)
        
        pred = torch.argmax(out,2)[:,indices]
        
        n_correct = torch.sum(pred == 0, dim=0)
        
        sum = sum + n_correct.cpu()

sum1 = sum
len1 = len(dataset)
print(str1, sum1)

##########################################################################################################################################################################################

dataset = torch.load(data_path2)

sum = torch.zeros(len(indices))

for sample in dataset:
    sample = torch.LongTensor(sample).to('cuda')

    with torch.no_grad():
        out, _, _ = model(sample)
        
        pred = torch.argmax(out,2)[:,indices]
        
        n_correct = torch.sum(pred == 1, dim=0)
        
        sum = sum + n_correct.cpu()

sum2 = sum
len2 = len(dataset)
print(str2, sum2)

##########################################################################################################################################################################################

dataset = torch.load(data_path3)

sum = torch.zeros(len(indices))

for sample in dataset:
    sample = torch.LongTensor(sample).to('cuda')

    with torch.no_grad():
        out, _, _ = model(sample)
        
        pred = torch.argmax(out,2)[:,indices]
        
        n_correct = torch.sum(pred == 2, dim=0)
        
        sum = sum + n_correct.cpu()

sum3 = sum
len3 = len(dataset)
print(str3, sum3)

##########################################################################################################################################################################################

total = sum1+sum2+sum3

print(len1,len2,len3,"\n")
    
for i in range(len(indices)):
    print("Accuracy at token", indices[i], "=", total[i].item()/(len1+len2+len3))
    

