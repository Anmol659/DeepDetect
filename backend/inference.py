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
        
        # Load YOUR trained weights
        if weights_path and os.path.exists(weights_path):
            logger.info(f"Loading YOUR trained model weights from {weights_path}")
            checkpoint = torch.load(weights_path, map_location=device)
            model.load_state_dict(checkpoint)
            logger.info("✓ YOUR trained model weights loaded successfully")
        else:
            raise FileNotFoundError(f"Your trained model not found at {weights_path}")
        
        model.eval()
        return model.to(device)
        
    except Exception as e:
        logger.error(f"Error building model: {e}")
        raise

# Model initialization - ONLY use your trained model
checkpoint_paths = [
    "checkpoints/best_model_3way.pth",
    "../checkpoints/best_model_3way.pth",
    "best_model_3way.pth"
]

model = None
for path in checkpoint_paths:
    if os.path.exists(path):
        try:
            model = build_efficientnet_b4(path)
            logger.info(f"✓ YOUR trained model loaded from {path}")
            break
        except Exception as e:
            logger.error(f"Failed to load model from {path}: {e}")
            continue

if model is None:
    logger.error("❌ YOUR trained model not found! Please ensure best_model_3way.pth exists")
    logger.error("Available files in checkpoints/:")
    if os.path.exists("checkpoints"):
        for file in os.listdir("checkpoints"):
            logger.error(f"  - {file}")
    raise FileNotFoundError("Your trained model (best_model_3way.pth) is required but not found")

# Image preprocessing - Enhanced for better detection
transform = transforms.Compose([
    transforms.Resize((380, 380)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Class labels - matching your training
index_to_label = {
    0: "ai_generated",
    1: "deepfake", 
    2: "real"
}

def validate_image(image_bytes):
    """Validate image format and content"""
    try:
        # Check if bytes are valid
        if not image_bytes or len(image_bytes) == 0:
            raise ValueError("Empty image data")
        
        # Try to open with PIL
        image = Image.open(BytesIO(image_bytes))
        
        # Verify it's a valid image
        image.verify()
        
        # Re-open for processing (verify closes the file)
        image = Image.open(BytesIO(image_bytes))
        
        # Check image properties
        if image.size[0] < 32 or image.size[1] < 32:
            raise ValueError("Image too small (minimum 32x32)")
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        return image
        
    except Exception as e:
        raise ValueError(f"Invalid image: {str(e)}")

def predict_image(image_bytes, use_tta=False):
    """
    Predict if an image is real, AI-generated, or deepfake using YOUR trained model
    
    Args:
        image_bytes: Raw image bytes
        use_tta: Test-time augmentation (ignored for compatibility)
    
    Returns:
        Dictionary with prediction results
    """
    try:
        if model is None:
            raise Exception("Your trained model is not loaded")
        
        # Validate and preprocess image
        image = validate_image(image_bytes)
        
        # Apply transforms
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
            "probabilities": class_probs,  # For extension compatibility
            "description": f"Classified as {predicted_label} with {confidence:.1%} confidence",
            "model_type": "your_trained_efficientnet_b4",
            "model_loaded": True
        }
        
        logger.info(f"Prediction: {predicted_label} ({confidence:.3f} confidence)")
        return result
        
    except Exception as e:
        logger.error(f"Error in image prediction: {e}")
        raise Exception(f"Image analysis failed: {str(e)}")

def predict_video(video_path, max_frames=5):
    """
    Predict video content by analyzing frames using YOUR trained model
    """
    try:
        if model is None:
            raise Exception("Your trained model is not loaded")
            
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            raise ValueError("Invalid video file or no frames found")
        
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
            "description": f"Video classified as {predicted_label}",
            "model_type": "your_trained_efficientnet_b4",
            "frames_analyzed": len(frame_predictions),
            "video_analysis": True,
            "model_loaded": True
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
        
        raise Exception(f"Video analysis failed: {str(e)}")

def get_model_info():
    """Get information about YOUR loaded model"""
    return {
        "model_type": "your_trained_efficientnet_b4" if model is not None else "not_loaded",
        "device": str(device),
        "classes": list(index_to_label.values()),
        "model_loaded": model is not None,
        "supports_video": model is not None,
        "training_info": "Custom trained EfficientNet-B4 for 3-class classification",
        "accuracy": "Based on your training results"
    }

# Verify model is loaded
if model is not None:
    logger.info("✓ YOUR trained EfficientNet-B4 model ready for inference")
else:
    logger.error("❌ YOUR trained model failed to load - check model file exists")