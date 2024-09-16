from utils.helper_functions import build_dataset
import argparse
import sys
def get_parser():
    description = 'Create a dataset.'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-d', '--source_dir', type=str, required=True, 
                        help='Path to the folder containing the images.')
    parser.add_argument('-f', '--dest_dir', type=str, required=True, 
                        help='Path to the folder where the dataset will be saved.')
    parser.add_argument('-v', '--verbose', action='store_true', 
                        help='Increase output verbosity')
    return parser

# Run the training
def run(args):
    build_dataset(args.source_dir, args.dest_dir, args.verbose)

if __name__ == '__main__':
    run(get_parser().parse_args(sys.argv[1:]))