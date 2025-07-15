import os
import time
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models, datasets
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import pandas as pd
from tqdm import tqdm

# ===========================================
# CONFIGURATION
# ===========================================
# Paths
csv_dataset_csv = r"C:/Users/anmol/Downloads/archive (1)/metadata.csv"
csv_dataset_root = r"C:/Users/anmol/Downloads/archive (1)/faces_224"
folder_dataset_train_dir = r"C:/Users/anmol/OneDrive/Desktop/Hackathon/DeepDetect/Datasets/train"
folder_dataset_val_dir = r"C:/Users/anmol/Downloads/archive/Dataset/Validation"
checkpoint_dir = r"C:/Users/anmol/OneDrive/Desktop/DeepDetect/checkpoints"

os.makedirs(checkpoint_dir, exist_ok=True)

# Training settings
batch_size = 8
num_epochs = 4
learning_rate = 1e-4
num_workers = 4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===========================================
# DATASETS
# ===========================================
class CSVDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_name = row["videoname"].replace(".mp4", ".jpg")
        label = 1 if row["label"] == "FAKE" else 0
        image_path = os.path.join(self.root_dir, image_name)
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

# Advanced augmentation
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(380, scale=(0.6, 1.0)),  # B4 resolution
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

# ===========================================
# MAIN
# ===========================================
if __name__ == "__main__":
    # Datasets
    csv_dataset = CSVDataset(csv_dataset_csv, csv_dataset_root, transform=train_transform)
    folder_dataset_train = datasets.ImageFolder(folder_dataset_train_dir, transform=train_transform)
    folder_dataset_val = datasets.ImageFolder(folder_dataset_val_dir, transform=val_transform)

    combined_train_dataset = ConcatDataset([csv_dataset, folder_dataset_train])

    # DataLoaders
    train_loader = DataLoader(
        combined_train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    val_loader = DataLoader(
        folder_dataset_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    # Model
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

    # Replace classifier head
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    model = model.to(device)

    # Loss & optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    # Training loop
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
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best_modelB4.pth"))
            print(" Saved new best model.")

        # Always save last epoch
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, "last_modelB4.pth"))

    print("Training complete!")
