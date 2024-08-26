import json
import random
import torch
import time
from open_lm.hf import *

from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# generates N random start tokens, from the same distribution as the starting tokens in file_path
def generate_start_tokens(tokenizer,N,file_path = 'shard_00000000_processed.jsonl'):
    # Read texts from file
    texts = []
    with open(file_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            texts.append(data['text'][:20])

    logging.info(f"Loaded {len(texts)} sequences from {file_path}")

    # Tokenize texts
    start_tokens = []
    for text in texts:
        input = tokenizer([text], return_tensors="pt")
        input = {k: v[:, :1] for k, v in input.items()}
        start_tokens.append(input)
    
    start_tokens = random.sample(start_tokens, N)
    return start_tokens


def generate_sequences(start_tokens,tokenizer,batch_size=16,max_new_tokens=800,output_file='output.jsonl'):

    model = AutoModelForCausalLM.from_pretrained("apple/DCLM-Baseline-7B")
    # Move model to GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Process inputs in batches
    for i in range(0, len(start_tokens), batch_size):
        batch = start_tokens[i:i + batch_size]
    
        # Concatenate tensors to create batch
        input_ids = torch.cat([item['input_ids'] for item in batch], dim=0).to(device)
        attention_mask = torch.cat([item['attention_mask'] for item in batch], dim=0).to(device)
    
        inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}
    
        # measure the time
        start = time.time()
        # Generate text for the batch
        gen_kwargs = {"max_new_tokens": max_new_tokens, "top_p": 0.8, "temperature": 0.99, "do_sample": True, "repetition_penalty": 1.1}
        output = model.generate(inputs['input_ids'], **gen_kwargs)
        output = output.cpu()
        end = time.time()
    
        # append output to jsonl file
        with open(output_file, 'a') as file:
            for generated_seq in output:
                decoded_output = tokenizer.decode(generated_seq.tolist(), skip_special_tokens=True)
                file.write(json.dumps({'text': decoded_output}) + '\n')

        # time taken in seconds
        time_taken = end - start
        tokens_generated = sum([len(seq) for seq in output])
        # log the time taken per batch
        
        logging.info(f"Tokens per second: {tokens_generated/time_taken}")



def main():
    # Create the argument parser
    parser = argparse.ArgumentParser(description='Generation script.')
    
    # Add arguments
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for generation')
    parser.add_argument('--num-seqs', type=int, default=128, help='Number of sequences to generate')
    parser.add_argument('--max-new-tokens', type=int, default=800, help='Maximal number of tokens to generate')
    parser.add_argument('--input-file', type=str, required=True, help='Path to the jsonl input file, for determining the distribution of the starting token')
    parser.add_argument('--output-file', type=str, required=True, help='Path to the jsonl output file, for writing the seqs to')
    
    # Parse the arguments
    args = parser.parse_args()
    logging.info(f"Arguments received: {args}")
        
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("apple/DCLM-Baseline-7B")
    
    start_tokens = generate_start_tokens(tokenizer,args.num_seqs,file_path = args.input_file)
    generate_sequences(start_tokens,tokenizer,batch_size=args.batch_size,max_new_tokens=args.max_new_tokens,output_file=args.output_file)

# Entry point of the script
if __name__ == "__main__":
    main()


