import argparse
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

def create_dataloaders(data_folder):
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load the dataset
    dataset = ImageFolder(root=data_folder, transform=transform)

    # Split dataset into train and validation
    val_split = 0.2
    num_samples = len(dataset)
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
