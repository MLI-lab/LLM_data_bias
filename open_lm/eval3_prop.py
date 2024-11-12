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
#parser.add_argument('--str3', type=str, help='test set 3')
#parser.add_argument('--str4', type=str, help='test set 4')
#parser.add_argument('--str5', type=str, help='test set 5')
#parser.add_argument('--str6', type=str, help='test set 6')
cmd_args = parser.parse_args()
args.model = cmd_args.model
args.classif_model_path = cmd_args.classif_model_path



###########################################################################################################

args.num_classes = 2


str1 = cmd_args.str1
str2 = cmd_args.str2
#str3 = cmd_args.str3
#str4 = cmd_args.str4
#str5 = cmd_args.str5
#str6 = cmd_args.str6


data1= "Llama1_gen"  #"DCLM_gen"
data2= "Dolma_gen"
data3= "FWEdu_gen"

base_path = '/media/datasets/test_set/'

data_path1 = base_path + data1 + '.pt'
data_path2 = base_path + data2 + '.pt'
data_path3 = base_path + data3 + '.pt'

model = test_classif_model(args)
model = model.to('cuda')


soft_max = torch.nn.Softmax(dim=2)
###########################################################################################################

pred = []
conf=[]

dataset = torch.load(data_path1)

for sample in dataset:
    sample = torch.LongTensor(sample).to('cuda')

    with torch.no_grad():
        out, _, _ = model(sample)
        
        out = soft_max(out)
        pred.append( torch.argmax(out,2)[:,-1].item() )
        conf.append( torch.max(out,2)[0][:,-1].item() )
        
c1 = pred.count(0)
c2 = pred.count(1)
#c3 = pred.count(2)
#c4 = pred.count(3)
#c5 = pred.count(4)
#c6 = pred.count(5)


sum_conf1 = sum(c for p, c in zip(pred, conf) if p == 0)
sum_conf2 = sum(c for p, c in zip(pred, conf) if p == 1)
#sum_conf3 = sum(c for p, c in zip(pred, conf) if p == 2)
#sum_conf4 = sum(c for p, c in zip(pred, conf) if p == 3)
#sum_conf5 = sum(c for p, c in zip(pred, conf) if p == 4)
#sum_conf6 = sum(c for p, c in zip(pred, conf) if p == 5)


av1 = sum_conf1/c1 if c1>0 else 0
av2 = sum_conf2/c2 if c2>0 else 0
#av3 = sum_conf3/c3 if c3>0 else 0
#av4 = sum_conf4/c4 if c4>0 else 0
#av5 = sum_conf5/c5 if c5>0 else 0
#av6 = sum_conf6/c6 if c6>0 else 0



length = len(dataset)

print(data1, ':')
print(str1, c1, "/", length, '=', c1/length, "with confidence ", av1)
print(str2, c2, "/", length, '=', c2/length, "with confidence ", av2)
#print(str3, c3, "/", length, '=', c3/length, "with confidence ", av3)
#print(str4, c4, "/", length, '=', c4/length, "with confidence ", av4)
#print(str5, c5, "/", length, '=', c5/length, "with confidence ", av5)
#print(str6, c6, "/", length, '=', c6/length, "with confidence ", av6)
print("\n")

exit()
##########################################################################################################################################################################################

pred = []
conf=[]

dataset = torch.load(data_path2)

for sample in dataset:
    sample = torch.LongTensor(sample).to('cuda')

    with torch.no_grad():
        out, _, _ = model(sample)
        
        out = soft_max(out)
        pred.append( torch.argmax(out,2)[:,-1].item() )
        conf.append( torch.max(out,2)[0][:,-1].item() )
        
c1 = pred.count(0)
c2 = pred.count(1)
c3 = pred.count(2)
c4 = pred.count(3)
c5 = pred.count(4)
c6 = pred.count(5)
c7 = pred.count(6)

sum_conf1 = sum(c for p, c in zip(pred, conf) if p == 0)
sum_conf2 = sum(c for p, c in zip(pred, conf) if p == 1)
sum_conf3 = sum(c for p, c in zip(pred, conf) if p == 2)
sum_conf4 = sum(c for p, c in zip(pred, conf) if p == 3)
sum_conf5 = sum(c for p, c in zip(pred, conf) if p == 4)
sum_conf6 = sum(c for p, c in zip(pred, conf) if p == 5)
sum_conf7 = sum(c for p, c in zip(pred, conf) if p == 6)

av1 = sum_conf1/c1 if c1>0 else 0
av2 = sum_conf2/c2 if c2>0 else 0
av3 = sum_conf3/c3 if c3>0 else 0
av4 = sum_conf4/c4 if c4>0 else 0
av5 = sum_conf5/c5 if c5>0 else 0
av6 = sum_conf6/c6 if c6>0 else 0
av7 = sum_conf7/c7 if c7>0 else 0


length = len(dataset)

print(data2, ':')
print(str1, c1, "/", length, '=', c1/length, "with confidence ", av1)
print(str2, c2, "/", length, '=', c2/length, "with confidence ", av2)
print(str3, c3, "/", length, '=', c3/length, "with confidence ", av3)
print(str4, c4, "/", length, '=', c4/length, "with confidence ", av4)
print(str5, c5, "/", length, '=', c5/length, "with confidence ", av5)
print(str6, c6, "/", length, '=', c6/length, "with confidence ", av6)
print(str7, c7, "/", length, '=', c7/length, "with confidence ", av7)
print("\n")

##########################################################################################################################################################################################

pred = []
conf=[]

dataset = torch.load(data_path3)

for sample in dataset:
    sample = torch.LongTensor(sample).to('cuda')

    with torch.no_grad():
        out, _, _ = model(sample)
        
        out = soft_max(out)
        pred.append( torch.argmax(out,2)[:,-1].item() )
        conf.append( torch.max(out,2)[0][:,-1].item() )
        
c1 = pred.count(0)
c2 = pred.count(1)
c3 = pred.count(2)
c4 = pred.count(3)
c5 = pred.count(4)
c6 = pred.count(5)
c7 = pred.count(6)

sum_conf1 = sum(c for p, c in zip(pred, conf) if p == 0)
sum_conf2 = sum(c for p, c in zip(pred, conf) if p == 1)
sum_conf3 = sum(c for p, c in zip(pred, conf) if p == 2)
sum_conf4 = sum(c for p, c in zip(pred, conf) if p == 3)
sum_conf5 = sum(c for p, c in zip(pred, conf) if p == 4)
sum_conf6 = sum(c for p, c in zip(pred, conf) if p == 5)
sum_conf7 = sum(c for p, c in zip(pred, conf) if p == 6)

av1 = sum_conf1/c1 if c1>0 else 0
av2 = sum_conf2/c2 if c2>0 else 0
av3 = sum_conf3/c3 if c3>0 else 0
av4 = sum_conf4/c4 if c4>0 else 0
av5 = sum_conf5/c5 if c5>0 else 0
av6 = sum_conf6/c6 if c6>0 else 0
av7 = sum_conf7/c7 if c7>0 else 0


length = len(dataset)

print(data3, ':')
print(str1, c1, "/", length, '=', c1/length, "with confidence ", av1)
print(str2, c2, "/", length, '=', c2/length, "with confidence ", av2)
print(str3, c3, "/", length, '=', c3/length, "with confidence ", av3)
print(str4, c4, "/", length, '=', c4/length, "with confidence ", av4)
print(str5, c5, "/", length, '=', c5/length, "with confidence ", av5)
print(str6, c6, "/", length, '=', c6/length, "with confidence ", av6)
print(str7, c7, "/", length, '=', c7/length, "with confidence ", av7)
print("\n")
##########################################################################################################################################################################################


