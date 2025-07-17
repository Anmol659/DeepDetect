from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
import tempfile
import os
from inference import predict_image, predict_video

app = Flask(__name__)
CORS(app)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/popup", methods=["GET"])
def popup():
    return send_from_directory('.', 'popup.html')

@app.route("/analyze", methods=["POST"])
def analyze():
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

    if not result or "label" not in result or "class_probs" not in result:
        return jsonify({"error": "Inference failed or returned incomplete data."}), 500

    return jsonify({
        "label": result["label"],
        "confidence": result["confidence"],
        "class_probs": result["class_probs"]
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
