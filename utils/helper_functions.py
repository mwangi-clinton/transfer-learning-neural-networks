import os
import shutil
from sklearn.model_selection import train_test_split
import torch
from torchvision import transforms
from PIL import Image

class_names = ['Acne', 'Carcinoma', 'Eczema', 'Keratosis', 'Milia', 'Rosacea']
def build_dataset(source_dir, dest_dir, verbose=False):
    # Create the validation and training folders in the destination folder
    train_dir = os.path.join(dest_dir, 'train')
    val_dir = os.path.join(dest_dir, 'val')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    for class_folder in os.listdir(source_dir):
        full_class_path = os.path.join(source_dir, class_folder)
        
        if not os.path.isdir(full_class_path):
            continue

        # Get all images in the current class folder
        image_files = [f for f in os.listdir(full_class_path) if os.path.isfile(os.path.join(full_class_path, f))]

        # Split the images into 80% train and 20% validation
        train_files, val_files = train_test_split(image_files, test_size=0.2, random_state=42)

        # Create directories for this class in train and val directories
        train_class_dir = os.path.join(train_dir, class_folder)
        val_class_dir = os.path.join(val_dir, class_folder)
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(val_class_dir, exist_ok=True)

        # Move train files to train directory
        for file in train_files:
            shutil.copy(os.path.join(full_class_path, file), os.path.join(train_class_dir, file))
            if verbose:
                print(f"Copied {file} to {train_class_dir}")

        # Move validation files to validation directory
        for file in val_files:
            shutil.copy(os.path.join(full_class_path, file), os.path.join(val_class_dir, file))
            if verbose:
                print(f"Copied {file} to {val_class_dir}")

    if verbose:
        print(f"Dataset created successfully in {dest_dir}")

    return dest_dir

def predict_image(image_path, model_path):
    # Load the saved model
    model = torch.load(model_path)
    model.eval()  # Set the model to evaluation mode

    # Define the same transforms used during training
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load and preprocess the image
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Add batch dimension

    # Make sure to use the same device as the model was trained on
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    image = image.to(device)

    # Make prediction
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    # Get the predicted class name
    predicted_class = class_names[predicted.item()]
    print(predicted_class)
    return predicted_class

