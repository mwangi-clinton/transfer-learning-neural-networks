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