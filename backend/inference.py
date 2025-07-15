import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import cv2
import os
from torchvision import transforms, models
from io import BytesIO

# ========== DEVICE ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== MODEL ==========
def build_efficientnet_b4(weights_path):
    weights = models.EfficientNet_B4_Weights.IMAGENET1K_V1
    model = models.efficientnet_b4(weights=weights)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 3)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model.to(device)

model = build_efficientnet_b4("checkpoints/best_model_3way.pth")

# ========== TRANSFORMS ==========
transform = transforms.Compose([
    transforms.Resize((380, 380)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ========== LABELS ==========
index_to_label = {
    0: "ai_generated",
    1: "deepfake",
    2: "real"
}

# ========== IMAGE INFERENCE ==========
def predict_image(image_bytes):
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    top_idx = int(np.argmax(probs))
    return {
        "label": index_to_label[top_idx],
        "confidence": float(np.max(probs)),
        "class_probs": {
            index_to_label[i]: float(p) for i, p in enumerate(probs)
        }
    }

# ========== VIDEO INFERENCE ==========
def predict_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < 5 and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        tensor = transform(img).unsqueeze(0).to(device)
        frames.append(tensor)
    cap.release()

    all_probs = []
    with torch.no_grad():
        for tensor in frames:
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
            all_probs.append(probs)

    mean_probs = np.mean(all_probs, axis=0)
    top_idx = int(np.argmax(mean_probs))

    # Optionally delete video
    if os.path.exists(video_path):
        os.remove(video_path)

    return {
        "label": index_to_label[top_idx],
        "confidence": float(np.max(mean_probs)),
        "class_probs": {
            index_to_label[i]: float(p) for i, p in enumerate(mean_probs)
        }
    }
