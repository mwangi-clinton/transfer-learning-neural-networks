import argparse
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from sklearn.model_selection import train_test_split
import os
import shutil

def  build_dataset(source_dir,dest_dir,verbose):
  #create the validation and traning folders in the destination folder
    train_dir = ''
    val_dir = ''
    os.makedirs(os.path.join(train_dir, dest_dir), exist_ok=True)
    os.makedirs(os.path.join(val_dir, dest_dir), exist_ok=True)
    for class_folder in os.listdir(source_dir):
        full_class_path = os.path.join(source_dir, class_folder)
    
    # Get all images in the current class folder
        image_files = [f for f in os.listdir(full_class_path) if os.path.isfile(os.path.join(full_class_path, f))]
    
    # Split the images into 80% train and 20% validation
        train_files, val_files = train_test_split(image_files, test_size=0.2, random_state=42)
    
    # Create directories for this class in train and val directories
        os.makedirs(os.path.join(train_dir, class_folder), exist_ok=True)
        os.makedirs(os.path.join(val_dir, class_folder), exist_ok=True)
    
    # Move train files to train directory
        for file in train_files:
            shutil.copy(os.path.join(full_class_path, file), os.path.join(train_dir, class_folder, file))
    
    # Move validation files to validation directory
        for file in val_files:
            shutil.copy(os.path.join(full_class_path, file), os.path.join(val_dir, class_folder, file))

 


    return dest_dir

