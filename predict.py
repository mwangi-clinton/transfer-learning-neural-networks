import torch
from torchvision import transforms
from PIL import Image
import argparse
from utils.helper_functions import predict_image
import sys

def get_parser():
    description = 'Create a dataset.'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-d', '--image_path', type=str, required=True,
                        help='Path to the folder containing the images.')
    parser.add_argument('-f', '--model_path', type=str, required=True,
                        help='Path to the folder where the dataset will be saved.')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Increase output verbosity')
    return parser

def run(args):
   predict_image(args.image_path, args.model_path, args.verbose)
   
