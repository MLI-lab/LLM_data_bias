# Data Preparation

All the datasets used in this work are publicly available on Hugging Face, the download links can be found in the [download_urls](https://github.com/MLI-lab/LLM_data_bias/blob/main/data_preparation/download_urls.md) file.

After downloading the dataset, extract the data into ``.jsonl`` files, which is the format the tokenizer expects. Place all ``.jsonl`` files of the same dataset in one folder. 

## Tokenization

 We tokenize the data using a BPE tokenizer and chunk it into shards. We use the [GPT-NeoX-20B](https://github.com/EleutherAI/gpt-neox) tokenizer. To tokenize and chunk the data, run the following from the root directory (LLM_data_bias):

```
    python open_lm/datapreprocess/make_2048.py \
    --input-files path_to_input_jsonl_data/*.jsonl
    --output-dir path_to_output_tokenized_data
    --num-workers 32
    --num-consumers 1
```  

This script tokenizes the data, and concatenates tokenized texts to form sequences of length 2048, which is the context length of the transformer. An <|endoftext|> token is added between the different texts. The sequences are divided into shards, where each shard has 8192 sequences. This means each shard has 8192 x 2048 = 16.78M tokens.

### Manifest file

By default the open_lm repoistory, picks shards to train on by sampling with replacement. To sample without replacement, which is recommended, we create a manifest file which specifies the shards to train on without replacement. This can be done by running the following from the root directory (LLM_data_bias):

```
python -m open_lm.utils.make_wds_manifest --data-dir path_to_output_tokenized_data
```

This will create a ``manifest.jsonl`` in the same directory as the tokenized data. This file is used in the training script to specify the path of the training data. 
