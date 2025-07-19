import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import cv2
import os
from torchvision import transforms, models
from io import BytesIO
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

def build_efficientnet_b4(weights_path):
    """Build EfficientNet-B4 model for 3-class classification"""
    try:
        # Load EfficientNet-B4 with ImageNet weights for feature extraction
        weights = models.EfficientNet_B4_Weights.IMAGENET1K_V1
        model = models.efficientnet_b4(weights=weights)
        
        # Replace classifier for 3 classes (ai_generated, deepfake, real)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 3)
        
        # Load your trained weights
        if weights_path and os.path.exists(weights_path):
            logger.info(f"Loading trained model weights from {weights_path}")
            checkpoint = torch.load(weights_path, map_location=device)
            model.load_state_dict(checkpoint)
            logger.info("✓ Trained model weights loaded successfully")
        else:
            logger.error(f"✗ Model checkpoint not found at {weights_path}")
            logger.error("Please ensure your trained model file exists at the specified path")
            return None
        
        model.eval()
        return model.to(device)
        
    except Exception as e:
        logger.error(f"Error building model: {e}")
        return None

# Model initialization - try multiple possible paths
checkpoint_paths = [
    "checkpoints/best_model_3way.pth",
    "../checkpoints/best_model_3way.pth",
    "backend/checkpoints/best_model_3way.pth",
    os.path.join(os.path.dirname(__file__), "..", "checkpoints", "best_model_3way.pth")
]

model = None
for checkpoint_path in checkpoint_paths:
    if os.path.exists(checkpoint_path):
        logger.info(f"Found checkpoint at: {checkpoint_path}")
        model = build_efficientnet_b4(checkpoint_path)
        if model is not None:
            break

if model is None:
    logger.error("✗ No trained model found. Please ensure your model checkpoint exists.")
    logger.error("Expected locations:")
    for path in checkpoint_paths:
        logger.error(f"  - {os.path.abspath(path)}")

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((380, 380)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Class labels - ensure this matches your training setup
index_to_label = {
    0: "ai_generated",
    1: "deepfake", 
    2: "real"
}

def predict_image(image_bytes):
    """
    Predict if an image is real, AI-generated, or deepfake using your trained model
    
    Args:
        image_bytes: Raw image bytes
    
    Returns:
        Dictionary with prediction results
    """
    try:
        if model is None:
            logger.error("Model not loaded - cannot make predictions")
            return {
                "label": "real",
                "confidence": 0.33,
                "confidence_level": "low",
                "class_probs": {
                    "ai_generated": 0.33,
                    "deepfake": 0.33,
                    "real": 0.34
                },
                "probabilities": {
                    "ai_generated": 0.33,
                    "deepfake": 0.33,
                    "real": 0.34
                },
                "description": "Model not loaded - using fallback",
                "model_type": "fallback",
                "error": "Trained model not available"
            }
            
        # Load and preprocess image with better format handling
        try:
            # First, try to open the image
            image = Image.open(BytesIO(image_bytes))
            
            # Handle different image formats
            if image.format not in ['JPEG', 'PNG', 'WEBP', 'BMP', 'TIFF']:
                logger.warning(f"Unsupported image format: {image.format}")
            
            # Convert to RGB (handles RGBA, grayscale, etc.)
            if image.mode in ['RGBA', 'LA']:
                # Create white background for transparent images
                background = Image.new('RGB', image.size, (255, 255, 255))
                if image.mode == 'RGBA':
                    background.paste(image, mask=image.split()[-1])  # Use alpha channel as mask
                else:
                    background.paste(image, mask=image.split()[-1])
                image = background
            elif image.mode != 'RGB':
                image = image.convert('RGB')
                
            # Validate image size
            if image.size[0] < 32 or image.size[1] < 32:
                raise ValueError("Image too small (minimum 32x32 pixels)")
                
            logger.info(f"Image processed: {image.size}, mode: {image.mode}")
            
        except Exception as img_error:
            logger.error(f"Image processing error: {img_error}")
            raise ValueError(f"Invalid image format or corrupted file: {str(img_error)}")
        
        tensor = transform(image).unsqueeze(0).to(device)
        
        # Make prediction with your trained model
        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        
        # Get results
        top_idx = int(np.argmax(probs))
        confidence = float(np.max(probs))
        predicted_label = index_to_label[top_idx]
        
        # Create class probabilities
        class_probs = {
            "ai_generated": float(probs[0]),
            "deepfake": float(probs[1]),
            "real": float(probs[2])
        }
        
        # Determine confidence level
        if confidence >= 0.8:
            confidence_level = "high"
        elif confidence >= 0.6:
            confidence_level = "medium"
        else:
            confidence_level = "low"
        
        result = {
            "label": predicted_label,
            "confidence": confidence,
            "confidence_level": confidence_level,
            "class_probs": class_probs,
            "probabilities": class_probs,
            "description": f"Classified as {predicted_label} with {confidence:.1%} confidence",
            "model_type": "efficientnet_b4_trained"
        }
        
        logger.info(f"Prediction: {predicted_label} ({confidence:.3f} confidence)")
        return result
        
    except Exception as e:
        logger.error(f"Error in image prediction: {e}")
        return {
            "label": "real",
            "confidence": 0.5,
            "confidence_level": "low",
            "class_probs": {
                "ai_generated": 0.2,
                "deepfake": 0.3,
                "real": 0.5
            },
            "probabilities": {
                "ai_generated": 0.2,
                "deepfake": 0.3,
                "real": 0.5
            },
            "description": "Analysis failed - fallback result",
            "model_type": "fallback",
            "error": str(e)
        }

def predict_video(video_path, max_frames=5):
    """
    Predict video content by analyzing frames using your trained model
    
    Args:
        video_path: Path to video file
        max_frames: Number of frames to analyze
    
    Returns:
        Dictionary with prediction results
    """
    try:
        if model is None:
            raise Exception("Trained model not loaded")
            
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Sample frames evenly
        if total_frames <= max_frames:
            frame_indices = list(range(total_frames))
        else:
            frame_indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
        
        frame_predictions = []
        
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                continue
                
            # Convert frame to PIL Image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            
            # Predict on frame
            tensor = transform(img).unsqueeze(0).to(device)
            
            with torch.no_grad():
                logits = model(tensor)
                probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
                frame_predictions.append(probs)
        
        cap.release()
        
        if not frame_predictions:
            raise ValueError("No frames could be processed")
        
        # Average predictions across frames
        avg_probs = np.mean(frame_predictions, axis=0)
        
        # Get final prediction
        top_idx = int(np.argmax(avg_probs))
        confidence = float(np.max(avg_probs))
        predicted_label = index_to_label[top_idx]
        
        class_probs = {
            "ai_generated": float(avg_probs[0]),
            "deepfake": float(avg_probs[1]),
            "real": float(avg_probs[2])
        }
        
        # Determine confidence level
        if confidence >= 0.8:
            confidence_level = "high"
        elif confidence >= 0.6:
            confidence_level = "medium"
        else:
            confidence_level = "low"
        
        result = {
            "label": predicted_label,
            "confidence": confidence,
            "confidence_level": confidence_level,
            "class_probs": class_probs,
            "probabilities": class_probs,
            "description": f"Video classified as {predicted_label} with {confidence:.1%} confidence",
            "model_type": "efficientnet_b4_trained",
            "frames_analyzed": len(frame_predictions),
            "video_analysis": True
        }
        
        logger.info(f"Video prediction: {predicted_label} ({confidence:.3f} confidence)")
        
        # Clean up video file
        if os.path.exists(video_path):
            try:
                os.remove(video_path)
            except:
                pass
        
        return result
        
    except Exception as e:
        logger.error(f"Error in video prediction: {e}")
        
        # Clean up video file
        if os.path.exists(video_path):
            try:
                os.remove(video_path)
            except:
                pass
        
        return {
            "label": "real",
            "confidence": 0.5,
            "confidence_level": "low",
            "class_probs": {
                "ai_generated": 0.2,
                "deepfake": 0.3,
                "real": 0.5
            },
            "probabilities": {
                "ai_generated": 0.2,
                "deepfake": 0.3,
                "real": 0.5
            },
            "description": "Video analysis failed - fallback result",
            "model_type": "fallback",
            "frames_analyzed": 0,
            "video_analysis": True,
            "error": str(e)
        }

def get_model_info():
    """Get information about the loaded model"""
    return {
        "model_type": "efficientnet_b4_trained" if model is not None else "none",
        "device": str(device),
        "classes": list(index_to_label.values()),
        "model_loaded": model is not None,
        "supports_video": model is not None,
        "architecture": "EfficientNet-B4",
        "num_classes": 3,
        "input_size": "380x380"
    }

# Log final status
if model is not None:
    logger.info("✓ Trained EfficientNet-B4 model ready for inference")
else:
    logger.warning("⚠ No trained model loaded - predictions will use fallback mode")