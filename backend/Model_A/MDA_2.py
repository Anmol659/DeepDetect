import os
import time
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models, datasets
from torch.utils.data import DataLoader
from tqdm import tqdm

# ===========================================
# CONFIGURATION
# ===========================================
# Paths
train_dir = r"C:/Users/anmol/OneDrive/Desktop/Hackathon/DeepDetect/Datasets/train"
val_dir = r"C:/Users/anmol/OneDrive/Desktop/Hackathon/DeepDetect/Datasets/Val"
checkpoint_dir = r"C:/Users/anmol/OneDrive/Desktop/DeepDetect/checkpoints"

os.makedirs(checkpoint_dir, exist_ok=True)

# Training settings
batch_size = 16
num_epochs = 6
learning_rate = 1e-4
num_workers = 4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    # ===========================================
    # DATASETS
    # ===========================================
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(380, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((380,380)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    print("Class to index mapping:", train_dataset.class_to_idx)

    # ===========================================
    # MODEL
    # ===========================================
    weights = models.EfficientNet_B4_Weights.IMAGENET1K_V1
    model = models.efficientnet_b4(weights=weights)

    # Freeze shallow layers
    for name, param in model.features.named_parameters():
        if any(name.startswith(f"{i}") for i in range(4)):
            param.requires_grad = False
        else:
            param.requires_grad = True

    # Unfreeze classifier
    for param in model.classifier.parameters():
        param.requires_grad = True

    # Replace classifier head for 3 classes
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 3)
    model = model.to(device)

    # ===========================================
    # LOSS & OPTIMIZER
    # ===========================================
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    # ===========================================
    # TRAINING LOOP
    # ===========================================
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Training]", leave=False)

        for images, labels in train_pbar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            train_pbar.set_postfix({
                "Loss": f"{running_loss / (total//batch_size + 1):.4f}",
                "Acc": f"{100 * correct / total:.2f}%"
            })

        train_acc = 100 * correct / total

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0

        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Validation]", leave=False)

        with torch.no_grad():
            for images, labels in val_pbar:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = 100 * val_correct / val_total
        epoch_time = time.time() - epoch_start

        print(f"\nEpoch {epoch+1}/{num_epochs} Completed - Train Acc: {train_acc:.2f}% - Val Acc: {val_acc:.2f}% - Time: {epoch_time/60:.2f} min\n")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best_model_3way.pth"))
            print(" Saved new best model.")

        torch.save(model.state_dict(), os.path.join(checkpoint_dir, "last_model_3way.pth"))

    print("Training complete!")
