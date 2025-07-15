import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import cv2
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from torchvision import transforms, models
from io import BytesIO
import logging
import traceback

app = Flask(__name__, static_folder="static")
CORS(app)  # Enable CORS for all routes

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Load model
def build_model(weights_path):
    logger.info(f"Loading model from: {weights_path}")
    weights = models.EfficientNet_B4_Weights.IMAGENET1K_V1
    model = models.efficientnet_b4(weights=weights)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 3)
    
    try:
        model.load_state_dict(torch.load(weights_path, map_location=device))
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise
    
    model.eval()
    return model.to(device)

MODEL_PATH = os.path.join("checkpoints", "best_model_3way.pth")

# Initialize model
try:
    model = build_model(MODEL_PATH)
    logger.info("Model initialization complete")
except Exception as e:
    logger.error(f"Failed to initialize model: {e}")
    model = None

transform = transforms.Compose([
    transforms.Resize((380, 380)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def classify_label(prob_array):
    """Classify based on probability array"""
    fake_prob = prob_array[0] + prob_array[1]
    real_prob = prob_array[2]
    if fake_prob > 0.8:
        return "fake"
    elif fake_prob > 0.5:
        return "possibly fake"
    else:
        return "real"

def predict_image(image_bytes):
    """Predict image authenticity"""
    try:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        logger.info(f"Image loaded successfully, size: {image.size}")
    except Exception as e:
        logger.error(f"Error loading image: {e}")
        raise ValueError("Invalid image file")
    
    # Transform and predict
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
    """Predict video authenticity by analyzing frames"""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < 5 and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (380, 380))  # Resize frame
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

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device)
    })

@app.route("/analyze", methods=["POST"])
def analyze():
    """Main analysis endpoint"""
    if model is None:
        return jsonify({"error": "Model not loaded. Please check server logs."}), 500
    
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400
    
    logger.info(f"Analyzing file: {file.filename}")

    try:
        filename = file.filename.lower()
        
        # Check file size (50MB limit)
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)
        
        if file_size > 50 * 1024 * 1024:  # 50MB
            return jsonify({"error": "File too large. Maximum size is 50MB."}), 400
        
        if filename.endswith((".jpg", ".jpeg", ".png", ".gif", ".webp")):
            file_data = file.read()
            result = predict_image(file_data)
            logger.info(f"Image analysis complete: {result['label']}")
            
        elif filename.endswith((".mp4", ".avi", ".mov", ".wmv", ".flv")):
            temp_path = f"temp_video_{os.getpid()}." + filename.split('.')[-1]
            try:
                file.save(temp_path)
                result = predict_video(temp_path)
                logger.info(f"Video analysis complete: {result['label']}")
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        else:
            return jsonify({"error": "Unsupported file type. Please upload JPG, PNG, MP4, AVI, or MOV files."}), 400
            
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500

    return jsonify(result)

# Serve index.html at /
@app.route("/")
def index():
    return app.send_static_file("index.html")

# Serve other static files (CSS/JS)
@app.route("/<path:path>")
def static_files(path):
    return send_from_directory("static", path)

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"Starting server on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)
