import yaml
import subprocess
import sys
import os

# Check if the YAML config path is passed as a command-line argument
if len(sys.argv) != 2:
    print("Usage: python pipeline_fasttext_train_and_eval.py <config.yaml>")
    sys.exit(1)

config_path = sys.argv[1]

# Load the configuration from the provided YAML file
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

# Define generated output file paths
train_output = "./data/" + config['name'] + "_train.csv"
test_output = "./data/" + config['name'] + "_test.csv"
model_output = config['name'] + "_model.bin"

# Prepare train data
subprocess.run([
    'python', 'prepare_classifier_data.py',
    '--reference', config['positive_train'],
    '--unlabeled', config['negative_train'],
    '--output_file', train_output
])

# Prepare test data
subprocess.run([
    'python', 'prepare_classifier_data.py',
    '--reference', config['positive_test'],
    '--unlabeled', config['negative_test'],
    '--output_file', test_output
])

# Train classifier
subprocess.run([
    'python', 'train_fasttext_classifer.py',
    '--input', train_output,
    '--valid_input', test_output,
    '--output', model_output
])

# Optionally delete the generated output files after the process is done
os.remove(train_output)
os.remove(test_output)
os.remove(model_output)