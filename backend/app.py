from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
import tempfile
import os
import logging

# Import with error handling
try:
    from inference import predict_image, predict_video, get_model_info
    MODEL_LOADED = True
    logging.info("✓ Enhanced inference module loaded successfully")
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
            "description": "Dummy prediction - install model for accuracy",
            "model_type": "dummy"
        }
    
    def predict_video(video_path):
        return predict_image(b"dummy")
    
    def get_model_info():
        return {
            "model_type": "dummy",
            "device": "cpu",
            "classes": ["ai_generated", "deepfake", "real"],
            "supports_tta": False,
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
        message = "DeepDetect API is running" if MODEL_LOADED else "API running with limited functionality (model not loaded)"
        
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

@app.route("/popup", methods=["GET"])
def popup():
    return send_from_directory('.', 'popup.html')

@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400
    
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        # Get content type, with fallback detection
        content_type = file.content_type or 'application/octet-stream'
        logger.info(f"Analyzing file: {file.filename} (type: {content_type})")

        # Read file data first to check if it's valid
        file_data = file.read()
        if len(file_data) == 0:
            return jsonify({"error": "Empty file"}), 400
        
        # Reset file pointer for potential re-reading
        file.seek(0)
        
        # Detect actual file type from file data if content_type is generic
        if content_type in ['binary/octet-stream', 'application/octet-stream']:
            # Try to detect image type from file signature
            if file_data.startswith(b'\xff\xd8\xff'):
                content_type = 'image/jpeg'
            elif file_data.startswith(b'\x89PNG\r\n\x1a\n'):
                content_type = 'image/png'
            elif file_data.startswith(b'GIF8'):
                content_type = 'image/gif'
            elif file_data.startswith(b'RIFF') and b'WEBP' in file_data[:12]:
                content_type = 'image/webp'
            elif file_data.startswith(b'\x00\x00\x00\x20ftypavif'):
                content_type = 'image/avif'
            else:
                # Try to open with PIL to verify it's an image
                try:
                    from PIL import Image
                    from io import BytesIO
                    Image.open(BytesIO(file_data)).verify()
                    content_type = 'image/unknown'  # Valid image but unknown format
                except Exception:
                    return jsonify({
                        "error": "Invalid file format",
                        "details": "File does not appear to be a valid image"
                    }), 400
        
        logger.info(f"Detected content type: {content_type}")
        # Accept images with any content type that we've verified
        if content_type.startswith("image/") or content_type == 'image/unknown':
            # Enhanced image analysis
            result = predict_image(file_data, use_tta=MODEL_LOADED)
            
        elif content_type.startswith("video/") and MODEL_LOADED:
            # Enhanced video analysis
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                file.save(tmp.name)
                result = predict_video(tmp.name, max_frames=10)
        else:
            supported_types = "images (JPG, PNG, WebP)"
            if MODEL_LOADED:
                supported_types += " and videos (MP4, AVI, MOV)"
            return jsonify({
                "error": f"Unsupported file type: {content_type}",
                "supported_types": supported_types,
                "details": "Please ensure you're uploading a valid image file"
            }), 400

        if not result or "label" not in result or "class_probs" not in result:
            return jsonify({
                "error": "Analysis failed or returned incomplete data",
                "details": "The model could not process this file"
            }), 500

        # Enhanced response with additional metadata
        return jsonify({
            "label": result["label"],
            "confidence": result["confidence"],
            "confidence_level": result.get("confidence_level", "unknown"),
            "class_probs": result["class_probs"],
            "probabilities": result.get("probabilities", result["class_probs"]),
            "description": result.get("description", result["label"]),
            "model_loaded": MODEL_LOADED,
            "model_type": result.get("model_type", "unknown"),
            "analysis_metadata": {
                "used_tta": result.get("used_tta", False),
                "frames_analyzed": result.get("frames_analyzed"),
                "temporal_consistency": result.get("temporal_consistency"),
                "video_analysis": result.get("video_analysis", False)
            }
        })
        
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        return jsonify({
            "error": f"Analysis failed: {str(e)}",
            "model_loaded": MODEL_LOADED,
            "details": "Please try again or contact support if the issue persists"
        }), 500

# New endpoint for batch analysis
@app.route("/analyze_batch", methods=["POST"])
def analyze_batch():
    """Analyze multiple files in a single request"""
    if not MODEL_LOADED:
        return jsonify({
            "error": "Batch analysis requires trained model",
            "details": "Please ensure the model is properly loaded"
        }), 503
    
    try:
        files = request.files.getlist("files")
        if not files:
            return jsonify({"error": "No files provided"}), 400
        
        results = []
        for i, file in enumerate(files[:10]):  # Limit to 10 files
            if file.filename == "":
                continue
                
            try:
                content_type = file.content_type
                if content_type.startswith("image/"):
                    result = predict_image(file.read(), use_tta=False)  # Disable TTA for batch
                    result["filename"] = file.filename
                    result["index"] = i
                    results.append(result)
            except Exception as e:
                results.append({
                    "filename": file.filename,
                    "index": i,
                    "error": str(e),
                    "label": "error"
                })
        
        return jsonify({
            "results": results,
            "total_processed": len(results),
            "model_loaded": MODEL_LOADED
        })
        
    except Exception as e:
        logger.error(f"Batch analysis error: {e}")
        return jsonify({
            "error": f"Batch analysis failed: {str(e)}",
            "model_loaded": MODEL_LOADED
        }), 500
if __name__ == "__main__":
    print("="*50)
    print("🚀 DeepDetect Enhanced Flask Server Starting...")
    print("="*50)
    if MODEL_LOADED:
        try:
            model_info = get_model_info()
            print(f"✓ Enhanced model loaded successfully")
            print(f"  Model type: {model_info['model_type']}")
            print(f"  Device: {model_info['device']}")
            print(f"  Classes: {', '.join(model_info['classes'])}")
            print(f"  TTA Support: {model_info.get('supports_tta', False)}")
            print(f"  Video Support: {model_info.get('supports_video', False)}")
        except:
            print("✓ Basic model loaded")
    else:
        print("⚠ Model not loaded - using fallback mode")
        print("  To get full functionality:")
        print("  1. Train the model using scripts in backend/Model_A/")
        print("  2. Or place trained model in checkpoints/ directory")
    print(f"Server will run on: http://localhost:5000")
    print("Features:")
    print("  • Enhanced UI with smooth animations")
    print("  • Improved deepfake detection accuracy")
    print("  • Test-time augmentation for better results")
    print("  • Video analysis with temporal consistency")
    print("  • Batch processing support")
    print("="*50)
    app.run(host="0.0.0.0", port=5000, debug=True)

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