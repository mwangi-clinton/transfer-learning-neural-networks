import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import timm
from utils.helper_functions import build_dataset
import os
from data_loader import create_dataloaders
from tqdm.auto import tqdm
import torchvision.models as models
NUMBER_OF_CLASSES = 6
num_epochs = 150

class CustomModel(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.5, freeze_layers=True):
        super(CustomModel, self).__init__()
        
        # Load the pre-trained ResNet50 model
        self.resnet50 = models.resnet50(pretrained=True)
        
        # Optionally freeze the early layers
        if freeze_layers:
            for param in self.resnet50.parameters():
                param.requires_grad = False
        
        # Replace the fully connected layer with a custom classifier
        # Original ResNet50 has a fully connected layer with 2048 input features and 1000 output classes
        # We modify this for our custom classification task
        num_features = self.resnet50.fc.in_features
        
        # Custom classifier with dropout and new output size
        self.resnet50.fc = nn.Sequential(
            nn.BatchNorm1d(num_features),         # BatchNorm for regularization
            nn.Dropout(dropout_rate),             # Dropout to prevent overfitting
            nn.Linear(num_features, 512),         # Hidden layer with 512 units
            nn.ReLU(),                            # ReLU activation
            nn.Linear(512, num_classes)           # Final output layer for num_classes
        )
        
    def forward(self, x):
        # Forward pass through the ResNet-50 model
        x = self.resnet50(x)
        return x



# Training function

def train_model(data_folder, model_folder, verbose):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create DataLoader
    train_loader, val_loader = create_dataloaders(data_folder)

    # Create model
    model = CustomModel(NUMBER_OF_CLASSES).to(device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.1)

    print("Initial model parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: mean = {param.data.mean():.4f}, std = {param.data.std():.4f}")

    # Training loop
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print('-' * 100)

        # Training phase
        model.train()
        running_train_loss = 0.0
        train_batches = 0

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()
            train_batches += 1

            print(f"Epoch {epoch+1}, Batch {batch_idx+1}: Training Loss: {loss.item():.4f}")

        epoch_train_loss = running_train_loss / train_batches

        # Validation phase
        model.eval()
        running_val_loss = 0.0
        val_batches = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(val_loader):
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                running_val_loss += loss.item()
                val_batches += 1

                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                print(f"Epoch {epoch+1}, Batch {batch_idx+1}: Validation Loss: {loss.item():.4f}")

        epoch_val_loss = running_val_loss / val_batches
        accuracy = 100. * correct / total

        print(f"Epoch {epoch+1} - Training Loss: {epoch_train_loss:.4f}, Validation Loss: {epoch_val_loss:.4f}, Accuracy: {accuracy:.2f}%")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")

        # Update learning rate
        scheduler.step(epoch_val_loss)

        # Print model parameters
        if epoch % 5 == 0:
            print("\nModel parameters:")
            for name, param in model.named_parameters():
                if param.requires_grad:
                    print(f"{name}: mean = {param.data.mean():.4f}, std = {param.data.std():.4f}")

        print()

    # Save the model
    os.makedirs(model_folder, exist_ok=True)
    model_path = os.path.join(model_folder, 'trained_model.pth')
    torch.save(model.state_dict(), model_path)

    if verbose:
        print(f"Model saved at {model_path}")


        
        
        
        
        
        