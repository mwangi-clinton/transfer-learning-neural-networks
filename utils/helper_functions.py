import argparse
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import os
import shutil

def  build_dataset(source_dir,dest_dir):
  #create the validation and traning folders in the destination folder
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




def create_dataloaders(data_folder):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load the dataset
    dataset = ImageFolder(root=data_folder, transform=transform)

    # Split dataset into train and validation
    val_split = 0.2
    num_samples = len(dataset)
    print(num_samples)
    val_size = int(num_samples * val_split)
    train_size = num_samples - val_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(dataset=val_dataset, batch_size=32, shuffle=False, num_workers=4)

    return train_loader, val_loader

def parse_args():
    parser = argparse.ArgumentParser(description='Train a model on a dataset and save it.')
    parser.add_argument('--data_folder', type=str, required=True, 
                        help='Path to the folder containing the dataset.')
    parser.add_argument('--save_path', type=str, required=True, 
                        help='Path to the folder where the trained model will be saved.')
    return parser.parse_args()
