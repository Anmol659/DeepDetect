from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
import tempfile
import os

# Import with error handling
try:
    from inference import predict_image, predict_video
    MODEL_LOADED = True
except Exception as e:
    print(f"Warning: Could not load inference module: {e}")
    MODEL_LOADED = False
    
    # Create dummy functions for testing
    def predict_image(image_bytes):
        return {
            "label": "real",
            "confidence": 0.85,
            "class_probs": {
                "ai_generated": 0.05,
                "deepfake": 0.10,
                "real": 0.85
            }
        }
    
    def predict_video(video_path):
        return predict_image(b"dummy")

app = Flask(__name__)
CORS(app)

# Health check endpoint for extension
@app.route("/health", methods=["GET"])
def health_check():
    status = "healthy" if MODEL_LOADED else "limited"
    message = "DeepDetect API is running" if MODEL_LOADED else "API running with limited functionality (model not loaded)"
    return jsonify({
        "status": status, 
        "message": message,
        "model_loaded": MODEL_LOADED
    }), 200

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/popup", methods=["GET"])
def popup():
    return send_from_directory('.', 'popup.html')

@app.route("/analyze", methods=["POST"])
def analyze():
    if not MODEL_LOADED:
        return jsonify({
            "error": "Model not loaded. Please check that the model checkpoint exists.",
            "details": "The trained model file 'best_model_3way.pth' was not found. Using fallback predictions."
        }), 503
    
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    content_type = file.content_type
    if content_type.startswith("image/"):
        result = predict_image(file.read())
    elif content_type.startswith("video/"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            file.save(tmp.name)
            result = predict_video(tmp.name)
    else:
        return jsonify({"error": "Unsupported file type"}), 400

    try:
        if not result or "label" not in result or "class_probs" not in result:
            return jsonify({"error": "Inference failed or returned incomplete data."}), 500

        return jsonify({
            "label": result["label"],
            "confidence": result["confidence"],
            "class_probs": result["class_probs"],
            "probabilities": result["class_probs"],  # Add this for extension compatibility
            "model_loaded": MODEL_LOADED
        })
    except Exception as e:
        return jsonify({
            "error": f"Analysis failed: {str(e)}",
            "model_loaded": MODEL_LOADED
        }), 500

if __name__ == "__main__":
    print("="*50)
    print("DeepDetect Flask Server Starting...")
    print("="*50)
    if MODEL_LOADED:
        print("✓ Model loaded successfully")
    else:
        print("⚠ Model not loaded - using fallback mode")
        print("  To get full functionality, ensure 'checkpoints/best_model_3way.pth' exists")
    print(f"Server will run on: http://localhost:5000")
    print("="*50)
    app.run(host="0.0.0.0", port=5000, debug=True)
