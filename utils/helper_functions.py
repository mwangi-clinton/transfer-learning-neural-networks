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

    trainTransform = transforms.Compose(
      [
        	transforms.RandomHorizontalFlip(),
	        transforms.RandomRotation(90),
          transforms.Resize((224, 224)),  # Resize images to 224x224 (or any size required by your model)
          transforms.ToTensor(),          # Convert image to tensor
          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalization values for pre-trained models
]
    )
    valTransform=([
      transforms.Resize((224,224)),
      transforms.ToTensor(),
      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load train and validation datasets
    train_dataset = ImageFolder(root=train_dir, transform=trainTransform)
    val_dataset = ImageFolder(root=val_dir, transform=valTransform)

    # Data loaders for train and validation datasets
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


    return train_loader, val_loader

