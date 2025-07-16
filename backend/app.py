from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ✅ Route for index.html
@app.route("/", methods=["GET"])
def index():
    return render_template("templates/index.html")

# ✅ Route for /analyze
@app.route("/analyze", methods=["POST"])
def analyze():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    return jsonify({
        "label": "real",
        "probabilities": {
            "ai_generated": 0.1,
            "deepfake": 0.2,
            "real": 0.7
        }
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
