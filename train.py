import argparse
import sys
from model import train_models

# Parse arguments
def get_parser():
    description = 'Train the Challenge models.'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-d', '--data_folder', type=str, required=True, 
                        help='Path to the folder containing the dataset.')
    parser.add_argument('-m', '--model_folder', type=str, required=True, 
                        help='Path to the folder where the trained model will be saved.')
    parser.add_argument('-v', '--verbose', action='store_true', 
                        help='Increase output verbosity')
    return parser

# Run the training
def run(args):
    train_models(args.data_folder, args.model_folder, args.verbose)

if __name__ == '__main__':
    run(get_parser().parse_args(sys.argv[1:]))
