import torch
from open_lm.params import parse_args
from open_lm.model import test_classif_model
import webdataset as wds
from open_lm.data import get_wds_dataset
from open_lm.data import sample_chunk

args = parse_args([])
args.per_gpu_val_batch_size = 8
args.vocab_size = 50432
args.seq_len = 2048
args.world_size = 1
args.rank = 0

args.model = "open_lm_160m"
model_path = "/media/logs/classif_C4160m3.2B_C4DCLM_320M/checkpoints/epoch_1.pt"

args.val_data = ['/media/datasets/C4/C4-shard-0000219.tar']

model = test_classif_model(args, model_path)
model = model.to('cuda')

dataset = get_wds_dataset(args, is_train=False, epoch=0, floor=True, tokenizer=None, data_key="txt", force_num_samples=None)

dataloader = dataset.dataloader

sum = 0
for sample in dataloader:
    (texts,) = sample
    texts = torch.LongTensor(texts).to('cuda')
    inputs, targets = sample_chunk(texts, args)
    
    with torch.no_grad():
        out, _, _ = model(inputs)
        
        pred = torch.argmax(out,2)[:,-1].sum()
        
        sum = sum + pred.item()

print(sum)
 

    

