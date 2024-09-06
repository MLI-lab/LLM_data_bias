# Description: This script is used to rewrite texts in a batch using OpenAI API.
# Batches are submitted and retrieved following https://platform.openai.com/docs/guides/batch/getting-started

# usage:
#   Submit a batch request, to re-write the texts in the input file:
#       python rewrite_texts_batch.py --submit c4_train_samples.json
#   Retrieve the batch results, and save them to the output file:  
#       python rewrite_texts_batch.py --retrieve batch_NsJSbrhKSRybSWfXKPmM52Ow --output-file c4_rewritten.jsonl
# 
# An openai account is required, and the API key should be set as an environment variable. 
# export OPENAI_API_KEY=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

from openai import OpenAI
import json
import argparse
import logging


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    # Create the argument parser
    parser = argparse.ArgumentParser(description='Rewriting script.')
    
    # Add arguments
    parser.add_argument('--retrieve', type=str, required=False, help='Path to the batch id to retrieve')
    parser.add_argument('--submit', type=str, required=False, help='Path to the jsonl input file, containt the texts to rewrite')
    parser.add_argument('--output-file', type=str, required=False, help='Path to the jsonl output file, for writing the rewritten texts to')

    # Parse the arguments
    args = parser.parse_args()
    logging.info(f"Arguments received: {args}")
        
    
    if args.submit:
        ### generate ..._batch.jsonl file for the batch request

        # Load JSON file
        with open(args.submit, "r") as f:
            data = [json.loads(line) for line in f]

        # obtain the part before .jsonl from input file
        batch_file_name = args.submit.split(".")[0] + "_batch.jsonl"

        with open(batch_file_name, 'w') as file:
            for i, entry in enumerate(data):
                # append output to jsonl file
                # {"custom_id": "request-1", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-3.5-turbo-0125", "messages": [{"role": "system", "content": "You are a helpful assistant."},{"role": "user", "content": "Hello world!"}],"max_tokens": 1000}}
                json_data = json.dumps({'custom_id': f"request-{i}", 
                                    'method': 'POST', 
                                    'url': '/v1/chat/completions', 
                                    'body': {'model': 'gpt-4o-mini', 
                                    'messages': [{'role': 'system', 'content': 'You are a helpful assistant.'}, 
                                                 {'role': 'user', 'content': "Rewrite this text while preserving the accuracy of its content: " + entry["text"] }], 
                                    'max_tokens': 2048}})
                # Write without adding newline at the end for the last item
                if i < len(data) - 1:
                    file.write(json_data + '\n')  # Add newline between entries
                else:
                    file.write(json_data)  # No newline for the last entry
            
        
        # 2. uploading batch input file
        client = OpenAI()

        batch_input_file = client.files.create(
            file=open(batch_file_name, "rb"),
            purpose="batch"
        )
        # 3. creating the batch
        batch_input_file_id = batch_input_file.id

        batch = client.batches.create(
            input_file_id=batch_input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={
                "description": "Rewrite texts in " + args.submit 
            }
        )

        print("submitted batch request with id " + batch.id)

    elif args.retrieve:
        
        # 4. checking the batch status
        client = OpenAI()
        batch = client.batches.retrieve(args.retrieve)
        print("Status: ", batch.status)
        print("Output file id: ", batch.output_file_id)
        print("Request counts: ", batch.request_counts)

        # 5. retrieving the results
        if batch.status == "completed":
            # 4. downloading the batch output file
            client = OpenAI()
            file_response = client.files.content(batch.output_file_id)
            # file_response is the content of the JSONL file, which is iterable
            
            # append output to jsonl file
            with open(args.output_file, 'w') as file:
                for line in file_response.text.split("\n"):
                    if line.strip():  # Skip empty lines
                        data = json.loads(line)
                        file.write(json.dumps({'text': data['response']['body']['choices'][0]['message']['content'] + "<|endoftext|>"}) + '\n')

# Entry point of the script
if __name__ == "__main__":
    main()
