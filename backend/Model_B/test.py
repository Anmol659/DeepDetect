import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import timm

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

import seaborn as sns
import matplotlib.pyplot as plt

# =================== CONFIG ===================
MODEL_PATH = 'checkpoints/best_model.pth'
TEST_CSV_PATH = r'C:/Users/anmol/OneDrive/Desktop/Hackathon/DeepDetect/Datasets/modelB/test.csv'
IMAGE_ROOT = r'C:/Users/anmol/OneDrive/Desktop/Hackathon/DeepDetect/Datasets/modelB/test_data_v2'
BATCH_SIZE = 16
IMG_SIZE = 224
NUM_CLASSES = 2
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SAVE_CSV = True
OUTPUT_CSV = 'test_predictions.csv'
# ==============================================

# ============ Dataset for Testing =============
class TestDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = os.path.join(IMAGE_ROOT, self.df.iloc[idx]['file_name'])
        label = int(self.df.iloc[idx]['label'])

        if not os.path.exists(img_path):
            print(f"[WARNING] Missing image: {img_path}")
            image = Image.new('RGB', (IMG_SIZE, IMG_SIZE))
        else:
            image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label

# ============== Load Test Data ================
df = pd.read_csv(TEST_CSV_PATH)

if 'file_name' not in df.columns or 'label' not in df.columns:
    raise ValueError("CSV must contain 'file_name' and 'label' columns.")

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

test_dataset = TestDataset(df, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# ============== Load Model ====================
model = timm.create_model('vit_large_patch16_224', pretrained=False, num_classes=NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# ============== Inference =====================
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Testing"):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        all_preds.extend(predicted.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

# ============== Metrics =======================
acc = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds)
recall = recall_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)

print(f"\n Accuracy:  {acc * 100:.2f}%")
print(f" Precision: {precision:.4f}")
print(f" Recall:    {recall:.4f}")
print(f" F1 Score:  {f1:.4f}")
print("\n Classification Report:")
print(classification_report(all_labels, all_preds, target_names=['Real', 'AI-Generated']))

# ============== Confusion Matrix ==============
cm = confusion_matrix(all_labels, all_preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Real', 'AI'], yticklabels=['Real', 'AI'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# ============== Save Predictions ==============
if SAVE_CSV:
    df['predicted'] = all_preds
    df.to_csv(OUTPUT_CSV, index=False)
    print(f" Predictions saved to: {OUTPUT_CSV}")
