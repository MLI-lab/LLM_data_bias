import fasttext
import argparse
import random
import os
import sys

def split_train_val(train_input, valid_frac):
    with open(train_input, "r") as f:
        lines = f.read().splitlines()
    
    random.shuffle(lines)

    num_train = int(len(lines) * (1-valid_frac))
    train_lines = lines[:num_train]
    valid_lines = lines[num_train:]

    out_base, file_ext = os.path.splitext(train_input)
    train_path = out_base + "_train" + file_ext
    valid_path = out_base + "_valid" + file_ext

    with open(train_path, 'w') as f:
        f.write("\n".join(train_lines))

    with open(valid_path, 'w') as f:
        f.write("\n".join(valid_lines))

    return train_path, valid_path

def print_results(N, p, r):
    print("Num samples\t" + str(N))
    print("Precision@{}\t{:.3f}".format(1, p))
    print("Recall@{}\t{:.3f}".format(1, r))

def get_args():
    # I/O parameters
    parser = argparse.ArgumentParser(description="Mix datasets in FastText format to train binary classifiers")
    parser.add_argument("--input", help="Training file for fasttext classifier (in fasttext format).", required=True)
    parser.add_argument("--valid_input", help='Data file for validating fasttext classifier', type=str)
    parser.add_argument("--valid_frac", help='Holdout ratio for validation', type=float, default=0)
    parser.add_argument("--output", help="Output path at which to save the model.")
    parser.add_argument("--seed", help="Random seed", type=int, default=42)

    # Hyperparameters 
    parser.add_argument("--lr", help='lr for fasttext classifier', type=float, default=0.1)
    parser.add_argument("--dim", help='size of word vectors', type=int, default=256)
    parser.add_argument("--ws", help='context window for fasttext classifier', type=int, default=5)
    parser.add_argument("--epoch", help='number of epochs to train fasttext classifier', type=int, default=5)
    parser.add_argument("--minCount", help='number of word occurrences needed to be included in vocab', type=int, default=1)
    parser.add_argument("--wordNgrams", help='maximum n-gram size to be included in vocab', type=int, default=1)

    # Parse the command-line arguments
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    random.seed(args.seed)
    train_input = args.input

    train_was_split = False
    if args.valid_input:
        valid_input = args.valid_input
    elif args.valid_frac > 0:
        train_input, valid_input = split_train_val(train_input, args.valid_frac)
        train_was_split = True
    else: 
        valid_input = None

    hyperparams = {'lr': args.lr, 'dim': args.dim, 'ws': args.ws, 'epoch': args.epoch, 'minCount': args.minCount, 'wordNgrams': args.wordNgrams}
    model = fasttext.train_supervised(input=train_input, **hyperparams)
  
    # Print the results
    print("Training results")
    print_results(*model.test(train_input))

    if valid_input is not None:
        print("Validation results")
        print_results(*model.test(valid_input)) 

    # Save the model
    if args.output:
        output = args.output if args.output.endswith(".bin") else args.output + ".bin"
        model.save_model(output)

    if train_was_split:
        os.remove(train_input)
        os.remove(valid_input)

if __name__ == "__main__":
    main()