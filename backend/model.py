import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import io
import numpy as np
import joblib

# Load your models
xception_model = torch.load("xception_best.pth", map_location=torch.device("cpu"))
xception_model.eval()

effnet_model = torch.load("last_model.pth", map_location=torch.device("cpu"))
effnet_model.eval()

meta_model = joblib.load("meta_classifier.pkl")

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def preprocess_image(file_bytes):
    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    return transform(image).unsqueeze(0)  # shape: (1, 3, 224, 224)

def predict_combined(file_bytes):
    img = preprocess_image(file_bytes)

    with torch.no_grad():
        deepfake_score = torch.sigmoid(xception_best(img)).item()
        ai_score = torch.sigmoid(last_model(img)).item()

    features = np.array([[deepfake_score, ai_score]])
    meta_pred = meta_model.predict(features)[0]
    confidence = float((deepfake_score + ai_score) / 2)

    if deepfake_score > 0.5 and ai_score > 0.5:
        label = "synthetic"
    elif deepfake_score > 0.5:
        label = "deepfake"
    elif ai_score > 0.5:
        label = "AI-generated"
    else:
        label = "real"

    return {
        "label": label,
        "confidence": round(confidence, 3),
        "scores": {
            "deepfake": round(deepfake_score, 3),
            "ai_generated": round(ai_score, 3)
        }
    }
