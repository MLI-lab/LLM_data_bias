# usage: python rewrite_texts.py --input-file c4_train_samples.json --output-file c4_train_samples_rewritten.jsonl
# export OPENAI_API_KEY=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
from openai import OpenAI
import json
import argparse
import logging
import time
import backoff
from openai import RateLimitError

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to retry the API call with exponential backoff
@backoff.on_exception(backoff.expo, RateLimitError, max_tries=5)
def completions_with_backoff(client, **kwargs):
    response = client.chat.completions.create(**kwargs)
    return response    

def main():
    # Create the argument parser
    parser = argparse.ArgumentParser(description='Rewriting script.')
    
    # Add arguments
    parser.add_argument('--input-file', type=str, required=True, help='Path to the jsonl input file, containt the texts to rewrite')
    parser.add_argument('--output-file', type=str, required=True, help='Path to the jsonl output file, for writing the rewritten texts to')

    # Parse the arguments
    args = parser.parse_args()
    logging.info(f"Arguments received: {args}")
        
    
    client = OpenAI()


    # Load JSON file
    with open(args.input_file, "r") as f:
        data = json.load(f)


    for i, entry in enumerate(data):
    
        # measure the time
        start = time.time()
        
        messages = [{"role": "system", "content": "You are a helpful assistant."}, 
                    {"role": "user", "content": "Rewrite this text while preserving the accuracy of its content:" + entry["text"]}
        ]
        
        try:
            # Call the completions_with_backoff function to handle rate limits
            #completion = completions_with_backoff(client, model="gpt-4o-mini", messages=messages,max_tokens=1000)
            completion = completions_with_backoff(client, model="gpt-3.5-turbo", messages=messages,max_tokens=1000)


            #completion = client.chat.completions.create(
            #    model="gpt-4o-mini",
            #    messages=messages
            #)

            print("-------- original text --------")
            print(entry["text"])
            print("-------- completion.choices --------")
            print(completion.choices[0].message.content + "<|endoftext|>")

            # append output to jsonl file
            with open(args.output_file, 'a') as file:
                file.write(json.dumps({'text': completion.choices[0].message.content + "<|endoftext|>"}) + '\n')
        
        except Exception as e:
            logging.error(f"An error occurred: {e}")
            continue

        end = time.time()
        # log the time and length of the generated text
        time_taken = end - start
        characters_generated = len(completion.choices[0].message.content)
        logging.info(f"Characters per second: {characters_generated/time_taken}")



# Entry point of the script
if __name__ == "__main__":
    main()
