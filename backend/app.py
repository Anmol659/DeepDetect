from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
import tempfile
import os
import logging
from PIL import Image
from io import BytesIO

# Import with error handling
try:
    from inference import predict_image, predict_video, get_model_info, model
    MODEL_LOADED = model is not None
    logging.info("✓ Inference module loaded successfully")
except Exception as e:
    logging.error(f"Could not load inference module: {e}")
    MODEL_LOADED = False
    
    # Create dummy functions for testing
    def predict_image(image_bytes):
        return {
            "label": "real",
            "confidence": 0.85,
            "confidence_level": "high",
            "class_probs": {
                "ai_generated": 0.05,
                "deepfake": 0.10,
                "real": 0.85
            },
            "probabilities": {
                "ai_generated": 0.05,
                "deepfake": 0.10,
                "real": 0.85
            },
            "description": "Dummy prediction - model not loaded",
            "model_type": "dummy"
        }
    
    def predict_video(video_path):
        return predict_image(b"dummy")
    
    def get_model_info():
        return {
            "model_type": "dummy",
            "device": "cpu",
            "classes": ["ai_generated", "deepfake", "real"],
            "model_loaded": False,
            "supports_video": False
        }

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Health check endpoint for extension
@app.route("/health", methods=["GET"])
def health_check():
    try:
        model_info = get_model_info()
        status = "healthy" if MODEL_LOADED else "limited"
        message = "DeepShield API is running" if MODEL_LOADED else "API running with limited functionality (model not loaded)"
        
        return jsonify({
            "status": status, 
            "message": message,
            "model_loaded": MODEL_LOADED,
            "model_info": model_info,
            "version": "2.0.0",
            "features": {
                "image_analysis": True,
                "video_analysis": MODEL_LOADED,
                "batch_processing": MODEL_LOADED,
                "confidence_levels": True
            }
        }), 200
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return jsonify({
            "status": "error",
            "message": f"Health check failed: {str(e)}",
            "model_loaded": False
        }), 500

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400
    
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        # Read file data
        file_data = file.read()
        if len(file_data) == 0:
            return jsonify({"error": "Empty file"}), 400
        
        logger.info(f"Analyzing file: {file.filename} (size: {len(file_data)} bytes)")

        # Validate image using PIL
        try:
            # Try to open and verify the image
            image = Image.open(BytesIO(file_data))
            image.verify()  # Verify it's a valid image
            
            # Re-open for processing (verify() closes the file)
            image = Image.open(BytesIO(file_data))
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            logger.info(f"Valid image: {image.size}, mode: {image.mode}")
            
            # Analyze image
            result = predict_image(file_data)
            
        except Exception as image_error:
            logger.error(f"Image validation failed: {image_error}")
            return jsonify({
                "error": "Invalid image file",
                "details": f"Could not process image: {str(image_error)}"
            }), 400

        # Validate result
        if not result or "label" not in result or "class_probs" not in result:
            logger.error("Analysis returned incomplete data")
            return jsonify({
                "error": "Analysis failed",
                "details": "The model could not process this file"
            }), 500

        # Return results
        response_data = {
            "label": result["label"],
            "confidence": result["confidence"],
            "confidence_level": result.get("confidence_level", "unknown"),
            "class_probs": result["class_probs"],
            "probabilities": result.get("probabilities", result["class_probs"]),
            "description": result.get("description", result["label"]),
            "model_loaded": MODEL_LOADED,
            "model_type": result.get("model_type", "unknown")
        }
        
        logger.info(f"Analysis successful: {result['label']} ({result['confidence']:.3f})")
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        return jsonify({
            "error": f"Analysis failed: {str(e)}",
            "model_loaded": MODEL_LOADED,
            "details": "Please try again or contact support if the issue persists"
        }), 500

# Model information endpoint
@app.route("/model_info", methods=["GET"])
def model_info():
    """Get detailed model information"""
    try:
        info = get_model_info()
        return jsonify({
            "model_info": info,
            "model_loaded": MODEL_LOADED,
            "api_version": "2.0.0"
        })
    except Exception as e:
        return jsonify({
            "error": f"Failed to get model info: {str(e)}",
            "model_loaded": MODEL_LOADED
        }), 500

if __name__ == "__main__":
    print("="*50)
    print("🚀 DeepShield Flask Server Starting...")
    print("="*50)
    if MODEL_LOADED:
        try:
            model_info = get_model_info()
            print(f"✓ Model loaded successfully")
            print(f"  Model type: {model_info['model_type']}")
            print(f"  Device: {model_info['device']}")
            print(f"  Classes: {', '.join(model_info['classes'])}")
            print(f"  Video Support: {model_info.get('supports_video', False)}")
        except:
            print("✓ Model loaded")
    else:
        print("⚠ Model not loaded - using fallback mode")
        print("  To get full functionality:")
        print("  1. Train the model using scripts in backend/Model_A/")
        print("  2. Or place trained model in checkpoints/ directory")
    print(f"Server will run on: http://localhost:5000")
    print("="*50)
    app.run(host="0.0.0.0", port=5000, debug=True)