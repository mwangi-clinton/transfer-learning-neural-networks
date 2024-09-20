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
NUMBER_OF_CLASSES = 6
num_epochs = 15


class CustomModel(nn.Module):
    num_classes = NUMBER_OF_CLASSES
    def __init__(self, num_classes):
        super(CustomModel, self).__init__()
        
        # Load the pre-trained EfficientNetB5 model
        self.model = timm.create_model('efficientnet_b5', pretrained=True)
        
        # Freeze all the layers in EfficientNetB5
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Get the number of features in the last layer
        if hasattr(self.model, 'classifier'):
            num_features = self.model.classifier.in_features
        elif hasattr(self.model, 'fc'):
            num_features = self.model.fc.in_features
        else:
            # If neither 'classifier' nor 'fc' exists, we need to investigate the model structure
            raise AttributeError("Model structure is different than expected. Please check the timm model definition.")
        
        # Replace the last layer with a new classifier
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.3),  # Add dropout for regularization
            nn.Linear(num_features, num_classes)
        )
        
        # Unfreeze the last few layers for fine-tuning
        for param in self.model.classifier.parameters():
            param.requires_grad = True
        
        # Optionally, unfreeze more layers if needed
        # for name, param in self.model.named_parameters():
        #     if "blocks.6" in name or "blocks.7" in name:  # Unfreeze the last two blocks
        #         param.requires_grad = True

    def forward(self, x):
        return self.model(x)


# Training function

def train_model(data_folder, model_folder, verbose):
    num_epochs=1
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
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1)

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


        
        
        
        
        
        