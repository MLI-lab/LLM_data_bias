import torch
from open_lm.params import parse_args
import argparse
from open_lm.model import test_classif_model
import webdataset as wds
from open_lm.data import get_wds_dataset
from open_lm.data import sample_chunk

args = parse_args([])
parser = argparse.ArgumentParser(description="Override params arguments with command-line arguments")
parser.add_argument('--model', type=str, help='Model name to use for evaluation')
parser.add_argument('--classif-model-path', type=str, help='Path to the classification model checkpoint')
parser.add_argument('--str1', type=str, help='test set 1')
parser.add_argument('--str2', type=str, help='test set 2')
parser.add_argument('--str3', type=str, help='test set 3')
parser.add_argument('--str4', type=str, help='test set 4')
cmd_args = parser.parse_args()
args.model = cmd_args.model
args.classif_model_path = cmd_args.classif_model_path


args.per_gpu_val_batch_size = 1
args.vocab_size = 50432
args.seq_len = 2047
args.world_size = 1
args.rank = 0

###########################################################################################################

args.num_classes = 4


str1 = cmd_args.str1
str2 = cmd_args.str2
str3 = cmd_args.str3
str4 = cmd_args.str4

data1= "DCLM"
data2= "Dolma"
data3= "FWEdu"

base_path = '/media/datasets/test_set/'

data_path1 = base_path + data1 + '.tar'
data_path2 = base_path + data2 + '.tar'
data_path3 = base_path + data3 + '.tar'

model = test_classif_model(args)
model = model.to('cuda')

###########################################################################################################

args.val_data = [data_path1]
dataset = get_wds_dataset(args, is_train=False, epoch=0, floor=True, tokenizer=None, data_key="txt", force_num_samples=None)
dataloader = dataset.dataloader


pred = []
for sample in dataloader:
    (texts,) = sample
    inputs = torch.LongTensor(texts).to('cuda')

    with torch.no_grad():
        out, _, _ = model(inputs)
        
        pred.append( torch.argmax(out,2)[:,-1].item() )
        
c1 = pred.count(0)
c2 = pred.count(1)
c3 = pred.count(2)
c4 = pred.count(3)

length = 4096

print(data1, ':')
print(str1, c1, "/", length, '=', c1/length)
print(str2, c2, "/", length, '=', c2/length)
print(str3, c3, "/", length, '=', c3/length)
print(str4, c4, "/", length, '=', c4/length)

##########################################################################################################################################################################################

args.val_data = [data_path2]
dataset = get_wds_dataset(args, is_train=False, epoch=0, floor=True, tokenizer=None, data_key="txt", force_num_samples=None)
dataloader = dataset.dataloader


pred = []
for sample in dataloader:
    (texts,) = sample
    inputs = torch.LongTensor(texts).to('cuda')

    with torch.no_grad():
        out, _, _ = model(inputs)
        
        pred.append( torch.argmax(out,2)[:,-1].item() )
        
c1 = pred.count(0)
c2 = pred.count(1)
c3 = pred.count(2)

length = 4096

print(data2, ':')
print(str1, c1, "/", length, '=', c1/length)
print(str2, c2, "/", length, '=', c2/length)
print(str3, c3, "/", length, '=', c3/length)

##########################################################################################################################################################################################

args.val_data =  [data_path3]
dataset = get_wds_dataset(args, is_train=False, epoch=0, floor=True, tokenizer=None, data_key="txt", force_num_samples=None)
dataloader = dataset.dataloader


pred = []
for sample in dataloader:
    (texts,) = sample
    inputs = torch.LongTensor(texts).to('cuda')

    with torch.no_grad():
        out, _, _ = model(inputs)
        
        pred.append( torch.argmax(out,2)[:,-1].item() )
        
c1 = pred.count(0)
c2 = pred.count(1)
c3 = pred.count(2)

length = 4096

print(data3, ':')
print(str1, c1, "/", length, '=', c1/length)
print(str2, c2, "/", length, '=', c2/length)
print(str3, c3, "/", length, '=', c3/length)

##########################################################################################################################################################################################


