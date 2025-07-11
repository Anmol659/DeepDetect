import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import timm

# ======================== CONFIG ========================
CSV_PATH = r'C:/Users/anmol/OneDrive/Desktop/Hackathon/DeepDetect/Datasets/modelB/train.csv'
IMAGE_ROOT = r'C:/Users/anmol/OneDrive/Desktop/Hackathon/DeepDetect/Datasets/modelB'
NUM_CLASSES = 2
EPOCHS = 10
BATCH_SIZE = 16
IMG_SIZE = 224
LR = 1e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# ========================================================

# ===================== CUSTOM DATASET ===================
class CustomDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = os.path.join(IMAGE_ROOT, self.df.iloc[idx]['file_name'])
        label = int(self.df.iloc[idx]['label'])

        if not os.path.exists(img_path):
            print(f"[WARNING] File not found: {img_path}")
            image = Image.new('RGB', (IMG_SIZE, IMG_SIZE))  # fallback black image
        else:
            image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)
        return image, label

# ======================= MAIN ===========================
def main():
    # Create checkpoints dir
    os.makedirs('checkpoints', exist_ok=True)

    # Load CSV
    print(f"Reading CSV from: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)

    if 'file_name' not in df.columns or 'label' not in df.columns:
        raise ValueError("CSV must contain 'file_name' and 'label' columns.")

    print(f"Total samples in CSV: {len(df)}")
    print("CSV preview:\n", df.head())

    # Train-val split
    train_df, val_df = train_test_split(df, test_size=0.1, stratify=df['label'], random_state=42)

    # Transforms
    train_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    val_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    # Datasets and Loaders
    train_dataset = CustomDataset(train_df, transform=train_transforms)
    val_dataset = CustomDataset(val_df, transform=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")
    print(f"Batch size: {BATCH_SIZE}, Device: {DEVICE}")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Model setup
    print("Loading Vision Transformer (ViT-Large)...")
    model = timm.create_model('vit_large_patch16_224', pretrained=True, num_classes=NUM_CLASSES)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scaler = torch.cuda.amp.GradScaler()

    best_acc = 0.0
    start_time = time.time()

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        epoch_start = time.time()

        print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            if (batch_idx + 1) % 10 == 0:
                print(f"[Batch {batch_idx+1}/{len(train_loader)}] Loss: {loss.item():.4f}")

        train_acc = 100 * correct / total

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        val_acc = 100 * correct / total

        # ETA & logging
        epoch_time = time.time() - epoch_start
        total_time = time.time() - start_time
        avg_epoch_time = total_time / (epoch + 1)
        eta = avg_epoch_time * (EPOCHS - epoch - 1)

        print(f"Epoch {epoch+1}: Train Acc = {train_acc:.2f}%, Val Acc = {val_acc:.2f}%, Time = {epoch_time:.1f}s, ETA = {eta:.1f}s")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'checkpoints/best_model.pth')
            print(" Best model saved!")

    # Save last model
    torch.save(model.state_dict(), 'checkpoints/last_model.pth')
    print(f"\n Training complete. Best Val Acc: {best_acc:.2f}%")

# ======================= RUN ============================
if __name__ == "__main__":
    main()
