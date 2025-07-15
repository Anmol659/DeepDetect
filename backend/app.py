import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import cv2
from flask import Flask, request, jsonify, send_from_directory
from torchvision import transforms, models
from io import BytesIO

app = Flask(__name__, static_folder="static")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
def build_model(weights_path):
    weights = models.EfficientNet_B4_Weights.IMAGENET1K_V1
    model = models.efficientnet_b4(weights=weights)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 3)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model.to(device)

MODEL_PATH = os.path.join("checkpoints", "best_model_3way.pth")
model = build_model(MODEL_PATH)

transform = transforms.Compose([
    transforms.Resize((380, 380)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def classify_label(prob_array):
    fake_prob = prob_array[0] + prob_array[1]
    real_prob = prob_array[2]
    if fake_prob > 0.8:
        return "fake"
    elif fake_prob > 0.5:
        return "possibly fake"
    else:
        return "real"

def predict_image(image_bytes):
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    label = classify_label(probs)
    return {
        "label": label,
        "probabilities": {
            "ai_generated": float(probs[0]),
            "deepfake": float(probs[1]),
            "real": float(probs[2])
        }
    }

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
    probs_list = []
    with torch.no_grad():
        for tensor in frames:
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
            probs_list.append(probs)
    avg_probs = np.mean(probs_list, axis=0)
    label = classify_label(avg_probs)
    return {
        "label": label,
        "probabilities": {
            "ai_generated": float(avg_probs[0]),
            "deepfake": float(avg_probs[1]),
            "real": float(avg_probs[2])
        }
    }

@app.route("/analyze", methods=["POST"])
def analyze():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    filename = file.filename.lower()
    if filename.endswith((".jpg", ".jpeg", ".png")):
        result = predict_image(file.read())
    elif filename.endswith((".mp4", ".avi", ".mov")):
        temp_path = "temp_video." + filename.split('.')[-1]
        file.save(temp_path)
        result = predict_video(temp_path)
        os.remove(temp_path)
    else:
        return jsonify({"error": "Unsupported file type"}), 400

    return jsonify(result)

# Serve index.html at /
@app.route("/")
def index():
    return app.send_static_file("index.html")

# Serve other static files (CSS/JS)
@app.route("/<path:path>")
def static_files(path):
    return send_from_directory("static", path)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
