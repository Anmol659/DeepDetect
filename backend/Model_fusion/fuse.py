import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import cv2

# Load Models
xception = torch.load("checkpoints/xception_best.pth", map_location="cpu")
vit = torch.load("checkpoints/last_model.pth", map_location="cpu")
xception.eval()
vit.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def preprocess(img_path):
    img = Image.open(img_path).convert("RGB")
    return transform(img).unsqueeze(0)

X, y = [], []

### --- Deepfake Dataset ---
deepfake_root = r"C:/Users/anmol/OneDrive/Desktop/Hackathon/DeepDetect/Datasets/test"
for label_folder in ["real", "fake"]:
    label = 0 if label_folder == "real" else 1
    folder = os.path.join(deepfake_root, label_folder)
    for fname in os.listdir(folder):
        path = os.path.join(folder, fname)
        try:
            img = preprocess(path)
            with torch.no_grad():
                d_score = torch.sigmoid(xception(img)).item()
                a_score = torch.sigmoid(vit(img)).item()
            X.append([d_score, a_score])
            y.append(label)
        except:
            continue

### --- AI-Generated Dataset ---
csv_path = r"C:/Users/anmol/OneDrive/Desktop/Hackathon/DeepDetect/Datasets/modelB/train.csv"
image_dir = r"C:/Users/anmol/OneDrive/Desktop/Hackathon/DeepDetect/Datasets/modelB/train_data"
df = pd.read_csv(csv_path)

for _, row in df.iterrows():
    img_id, label = row['id'], row['label']
    path = os.path.join(image_dir, f"{img_id}.jpg")
    if not os.path.exists(path):
        continue
    try:
        img = preprocess(path)
        with torch.no_grad():
            d_score = torch.sigmoid(xception(img)).item()
            a_score = torch.sigmoid(vit(img)).item()
        X.append([d_score, a_score])
        y.append(label)
    except:
        continue

# Train Meta Classifier
X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

meta = LogisticRegression()
meta.fit(X_train, y_train)

print("[✔] Meta-Classifier Results")
print("Accuracy:", meta.score(X_test, y_test))
print(classification_report(y_test, meta.predict(X_test)))

joblib.dump(meta, "meta_classifier.pkl")
print("[✔] Saved: meta_classifier.pkl")
