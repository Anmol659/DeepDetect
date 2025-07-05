from flask import Flask, request, jsonify
import torch
from torchvision import transforms
from PIL import Image
from model import get_xception_model

# Flask app
app = Flask(__name__)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = get_xception_model(pretrained=False)
model.load_state_dict(torch.load("checkpoints/xception_best.pth", map_location=device))
model = model.to(device)
model.eval()

# Classes
classes = ["Fake", "Real"]

# Transform (same as validation)
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])

@app.route("/")
def index():
    return "DeepDetect Model API"

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files["image"]
    try:
        img = Image.open(file).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Invalid image: {str(e)}"}), 400

    # Preprocess
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, pred_idx = torch.max(probs, 1)

    result = {
        "predicted_class": classes[pred_idx.item()],
        "confidence": round(confidence.item(), 4)
    }

    return jsonify(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
