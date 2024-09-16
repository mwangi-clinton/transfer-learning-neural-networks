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
NUMBER_OF_CLASSES = 6

class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.num_classes = NUMBER_OF_CLASSES
        # Load the pre-trained EfficientNetB5 model
        self.model = timm.create_model('efficientnet_b5', pretrained=True)
        
        # # Remove the FC layers (last two layers) from EfficientNetB5
        # self.model = nn.Sequential(*list(self.model.children())[:-2])
        
        # # New classifier with softmax (logistic regression)
        # self.classifier = nn.Sequential(
        #     nn.AdaptiveAvgPool2d(1),  # Pooling layer to reduce the spatial dimensions
        #     nn.Flatten(),             # Flatten the pooled output to 1D
        #     nn.Linear(2048, self.num_classes),  # New fully connected layer with num_classes output
        #     nn.Softmax(dim=1)         # Softmax for multi-class classification
        # )

        # Freeze all the layers in EfficientNetB5
        for param in self.model.parameters():
            param.requires_grad = False
        modelOyutputFeatures = self.model.fc.in_features
        self.model.fc = nn.Linear(modelOyutputFeatures,self.num_classes)

        # Optionally, unfreeze the last few layers for fine-tuning (optional)
        # for param in list(self.model[-1].parameters()):
        #     param.requires_grad = True

    # def forward(self, x):
    #     x = self.model(x)   # Extract features from the pre-trained EfficientNetB5
    #     x = self.classifier(x)  # Classify based on the extracted features
    #     return x


# Training function
def train_models(data_folder, model_folder, verbose):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create DataLoader
    train_loader, val_loader = create_dataloaders(data_folder)

    # Create model
    model = CustomModel().to(device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Training loop
    num_epochs = 10  # Modify as needed
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

        if verbose:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

    # Validation loop (optional)
    model.eval()
    with torch.no_grad():
        val_loss = 0.0
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

        if verbose:
            print(f"Validation Loss: {val_loss/len(val_loader):.4f}")

    # Save the model
    os.makedirs(model_folder, exist_ok=True)
    model_path = os.path.join(model_folder, 'trained_model.pth')
    torch.save(model.state_dict(), model_path)
    
    if verbose:
        print(f"Model saved at {model_path}")