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
        # Load pretrained EfficientNet-B4
        weights = models.EfficientNet_B4_Weights.IMAGENET1K_V1
        model = models.efficientnet_b4(weights=weights)
        
        # Replace classifier for 3 classes (ai_generated, deepfake, real)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 3)
        
        # Load trained weights if available
        if weights_path and os.path.exists(weights_path):
            logger.info(f"Loading model weights from {weights_path}")
            model.load_state_dict(torch.load(weights_path, map_location=device))
            logger.info("✓ Model weights loaded successfully")
        else:
            logger.warning("⚠ No trained weights found, using ImageNet pretrained + random classifier")
            # Initialize the new classifier layer
            nn.init.xavier_uniform_(model.classifier[1].weight)
            nn.init.zeros_(model.classifier[1].bias)
        
        model.eval()
        return model.to(device)
        
    except Exception as e:
        logger.error(f"Error building model: {e}")
        raise

# Model initialization
checkpoint_path = "checkpoints/best_model_3way.pth"
try:
    model = build_efficientnet_b4(checkpoint_path)
    logger.info("✓ Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    model = None

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((380, 380)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Class labels
index_to_label = {
    0: "ai_generated",
    1: "deepfake", 
    2: "real"
}

def predict_image(image_bytes, use_tta=False):
    """
    Predict if an image is real, AI-generated, or deepfake
    
    Args:
        image_bytes: Raw image bytes
        use_tta: Ignored for compatibility
    
    Returns:
        Dictionary with prediction results
    """
    try:
        if model is None:
            raise Exception("Model not loaded")
            
        # Load and preprocess image
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        tensor = transform(image).unsqueeze(0).to(device)
        
        # Make prediction
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
            "description": f"Classified as {predicted_label}",
            "model_type": "efficientnet_b4"
        }
        
        logger.info(f"Prediction: {predicted_label} ({confidence:.3f} confidence)")
        return result
        
    except Exception as e:
        logger.error(f"Error in image prediction: {e}")
        # Return fallback result
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
    Predict video content by analyzing frames
    
    Args:
        video_path: Path to video file
        max_frames: Number of frames to analyze
    
    Returns:
        Dictionary with prediction results
    """
    try:
        if model is None:
            raise Exception("Model not loaded")
            
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
        
        # Average predictions
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
            "description": f"Video classified as {predicted_label}",
            "model_type": "efficientnet_b4",
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
        
        # Return fallback result
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
        "model_type": "efficientnet_b4" if model is not None else "fallback",
        "device": str(device),
        "classes": list(index_to_label.values()),
        "model_loaded": model is not None,
        "supports_video": model is not None
    }

logger.info("✓ Inference module loaded")
if model is not None:
    logger.info("✓ EfficientNet-B4 model ready")
else:
    logger.warning("⚠ Model not loaded - using fallback mode")