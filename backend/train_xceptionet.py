import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
import os
import time

def main():
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomRotation(10),
        transforms.RandomApply([transforms.GaussianBlur(3)], p=0.3),
        transforms.ToTensor()
    ])

    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])

    # Datasets
    train_dir = r"C:\Users\anmol\OneDrive\Desktop\Hackathon\DeepDetect\Datasets\train"
    val_dir = r"C:\Users\anmol\OneDrive\Desktop\Hackathon\DeepDetect\Datasets\val"

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)

    # DataLoaders (use num_workers=0 for Windows safety)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

    # Model
    model = timm.create_model('xception', pretrained=True)
    model.reset_classifier(num_classes=2)
    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(model.get_classifier().in_features, 2)
    )
    model = model.to(device)

    # Loss
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=0.0003)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

    # Training parameters
    num_epochs = 10
    best_val_loss = float("inf")
    early_stop_counter = 0
    early_stop_patience = 5
    start_time = time.time()

    for epoch in range(num_epochs):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = 100 * correct / total

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (preds == labels).sum().item()

        val_epoch_loss = val_loss / val_total
        val_epoch_acc = 100 * val_correct / val_total

        epoch_time = time.time() - epoch_start
        total_time = time.time() - start_time
        avg_epoch_time = total_time / (epoch + 1)
        remaining_epochs = num_epochs - (epoch + 1)
        estimated_remaining = remaining_epochs * avg_epoch_time

        print(f"\nEpoch [{epoch+1}/{num_epochs}]")
        print(f"Train Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.2f}%")
        print(f"Validation Loss: {val_epoch_loss:.4f} | Accuracy: {val_epoch_acc:.2f}%")
        print(f"Time for this epoch: {epoch_time:.2f} sec")
        print(f"Elapsed time: {total_time/60:.2f} min")
        print(f"Estimated remaining time: {estimated_remaining/60:.2f} min")

        scheduler.step(val_epoch_loss)

        if val_epoch_loss < best_val_loss:
            print("Validation loss improved. Saving model...")
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), "checkpoints/xception_best.pth")
            best_val_loss = val_epoch_loss
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= early_stop_patience:
                print("Early stopping triggered.")
                break

if __name__ == "__main__":
    main()
