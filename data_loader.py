import os
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch

def create_dataloaders(dataset_path, verbose=True ):
    batch_size=64
    num_workers=2
    # Define the transforms
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Paths for training and validation data
    train_path = os.path.join(dataset_path, 'train')
    val_path = os.path.join(dataset_path, 'val')

    # Create datasets
    train_dataset = ImageFolder(train_path, transform=train_transform)
    val_dataset = ImageFolder(val_path, transform=val_transform)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    #You can use this code to check the output of the loaders
    """# Print information about the datasets
    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of validation samples: {len(val_dataset)}")
    print(f"Classes: {train_dataset.classes}")

    # Print labels for the first batch of validation data
    val_iter = iter(train_loader)
    first_batch = next(val_iter)
    print("Labels for the first batch of validation data:")
    print(first_batch[1])  # Print the labels"""

    return train_loader, val_loader