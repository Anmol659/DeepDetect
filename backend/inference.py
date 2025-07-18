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

# ========== DEVICE ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# ========== ENHANCED MODEL ARCHITECTURE ==========
class EnhancedDeepfakeDetector(nn.Module):
    """Enhanced model combining EfficientNet-B4 with attention mechanisms"""
    
    def __init__(self, num_classes=3, dropout_rate=0.3):
        super(EnhancedDeepfakeDetector, self).__init__()
        
        # Base EfficientNet-B4
        self.backbone = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1)
        
        # Remove the original classifier
        backbone_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=backbone_features,
            num_heads=8,
            dropout=dropout_rate,
            batch_first=True
        )
        
        # Enhanced classifier with residual connections
        self.classifier = nn.Sequential(
            nn.LayerNorm(backbone_features),
            nn.Dropout(dropout_rate),
            nn.Linear(backbone_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.3),
            nn.Linear(256, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize classifier weights"""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # Extract features from backbone
        features = self.backbone(x)
        
        # Reshape for attention (batch_size, seq_len=1, features)
        features_reshaped = features.unsqueeze(1)
        
        # Apply attention
        attended_features, _ = self.attention(
            features_reshaped, features_reshaped, features_reshaped
        )
        
        # Squeeze back to original shape
        attended_features = attended_features.squeeze(1)
        
        # Residual connection
        enhanced_features = features + attended_features
        
        # Final classification
        output = self.classifier(enhanced_features)
        
        return output

def build_enhanced_model(weights_path=None):
    """Build enhanced deepfake detection model"""
    try:
        model = EnhancedDeepfakeDetector(num_classes=3, dropout_rate=0.3)
        
        if weights_path and os.path.exists(weights_path):
            logger.info(f"Loading model weights from {weights_path}")
            state_dict = torch.load(weights_path, map_location=device)
            model.load_state_dict(state_dict)
            logger.info("✓ Enhanced model loaded successfully")
        else:
            logger.warning("⚠ No trained weights found, using pretrained backbone only")
            logger.info("For better accuracy, train the model using the training scripts")
        
        model.eval()
        return model.to(device)
        
    except Exception as e:
        logger.error(f"Error building enhanced model: {e}")
        # Fallback to basic model
        return build_basic_efficientnet(weights_path)

def build_basic_efficientnet(weights_path=None):
    """Fallback to basic EfficientNet-B4 model"""
    try:
        logger.info("Building basic EfficientNet-B4 model")
        weights = models.EfficientNet_B4_Weights.IMAGENET1K_V1
        model = models.efficientnet_b4(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 3)
        
        if weights_path and os.path.exists(weights_path):
            try:
                model.load_state_dict(torch.load(weights_path, map_location=device))
                logger.info("✓ Basic model weights loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load weights: {e}, using ImageNet pretrained")
                # Initialize the new classifier layer
                nn.init.xavier_uniform_(model.classifier[1].weight)
                nn.init.zeros_(model.classifier[1].bias)
        else:
            logger.info("Using ImageNet pretrained weights with random classifier")
            nn.init.xavier_uniform_(model.classifier[1].weight)
            nn.init.zeros_(model.classifier[1].bias)
        
        model.eval()
        return model.to(device)
        
    except Exception as e:
        logger.error(f"Error building basic model: {e}")
        raise

# ========== MODEL INITIALIZATION ==========
# Try different possible checkpoint paths
checkpoint_paths = [
    "checkpoints/enhanced_model.pth",
    "checkpoints/best_model_3way.pth",
    "../checkpoints/enhanced_model.pth",
    "../checkpoints/best_model_3way.pth",
    "enhanced_model.pth",
    "best_model_3way.pth"
]

model = None
model_type = "fallback"

# Try to load enhanced model first
for path in checkpoint_paths:
    if os.path.exists(path):
        try:
            if "enhanced" in path:
                model = build_enhanced_model(path)
                model_type = "enhanced"
                logger.info(f"✓ Enhanced model loaded from {path}")
            else:
                model = build_enhanced_model(path)  # Try enhanced architecture with basic weights
                model_type = "enhanced_with_basic_weights"
                logger.info(f"✓ Enhanced architecture with basic weights from {path}")
            break
        except Exception as e:
            logger.warning(f"Failed to load model from {path}: {e}")
            continue

# Fallback to basic model if enhanced model fails
if model is None:
    logger.info("Loading fallback model with ImageNet weights")
    model = build_basic_efficientnet()
    model_type = "basic_fallback"

logger.info(f"Model type: {model_type}")

# ========== ENHANCED TRANSFORMS ==========
# Training-time augmentation for better robustness
train_transform = transforms.Compose([
    transforms.Resize((380, 380)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=5),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Standard inference transform
inference_transform = transforms.Compose([
    transforms.Resize((380, 380)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ========== ENHANCED LABELS ==========
index_to_label = {
    0: "ai_generated",
    1: "deepfake", 
    2: "real"
}

label_descriptions = {
    "ai_generated": "AI-Generated Content",
    "deepfake": "Deepfake/Face-Swapped",
    "real": "Authentic Content"
}

confidence_thresholds = {
    "high": 0.8,
    "medium": 0.6,
    "low": 0.4
}

# ========== ENHANCED IMAGE INFERENCE ==========
def predict_image(image_bytes, use_tta=True):
    """
    Enhanced image prediction with Test Time Augmentation (TTA)
    
    Args:
        image_bytes: Raw image bytes
        use_tta: Whether to use test time augmentation for better accuracy
    
    Returns:
        Dictionary with prediction results
    """
    try:
        # Validate input
        if not image_bytes or len(image_bytes) == 0:
            raise ValueError("Empty image data provided")
        
        logger.info(f"Processing image: {len(image_bytes)} bytes")
        
        # Load and preprocess image
        try:
            image = Image.open(BytesIO(image_bytes)).convert("RGB")
            logger.info(f"Image loaded successfully: {image.size}")
        except Exception as e:
            logger.error(f"Failed to load image: {e}")
            raise ValueError(f"Invalid image format: {str(e)}")
        
        # Validate image size
        if image.size[0] < 32 or image.size[1] < 32:
            raise ValueError("Image too small (minimum 32x32 pixels)")
        
        # Basic prediction
        tensor = inference_transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            if use_tta and model_type in ["enhanced", "enhanced_with_basic_weights"]:
                # Test Time Augmentation for better accuracy
                predictions = []
                
                # Original image
                logits = model(tensor)
                predictions.append(torch.softmax(logits, dim=1))
                
                # Horizontally flipped
                flipped_tensor = torch.flip(tensor, dims=[3])
                logits_flipped = model(flipped_tensor)
                predictions.append(torch.softmax(logits_flipped, dim=1))
                
                # Slightly rotated versions
                for angle in [-2, 2]:
                    rotated_image = transforms.functional.rotate(image, angle)
                    rotated_tensor = inference_transform(rotated_image).unsqueeze(0).to(device)
                    logits_rotated = model(rotated_tensor)
                    predictions.append(torch.softmax(logits_rotated, dim=1))
                
                # Average predictions
                avg_probs = torch.mean(torch.stack(predictions), dim=0)[0].cpu().numpy()
            else:
                # Standard single prediction
                logits = model(tensor)
                avg_probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        
        # Get prediction results
        top_idx = int(np.argmax(avg_probs))
        confidence = float(np.max(avg_probs))
        predicted_label = index_to_label[top_idx]
        
        # Determine confidence level
        if confidence >= confidence_thresholds["high"]:
            confidence_level = "high"
        elif confidence >= confidence_thresholds["medium"]:
            confidence_level = "medium"
        else:
            confidence_level = "low"
        
        # Create detailed class probabilities
        class_probs = {
            index_to_label[i]: float(p) for i, p in enumerate(avg_probs)
        }
        
        # Enhanced result with additional metadata
        result = {
            "label": predicted_label,
            "confidence": confidence,
            "confidence_level": confidence_level,
            "class_probs": class_probs,
            "probabilities": class_probs,  # For backward compatibility
            "description": label_descriptions.get(predicted_label, predicted_label),
            "model_type": model_type,
            "used_tta": use_tta and model_type in ["enhanced", "enhanced_with_basic_weights"]
        }
        
        logger.info(f"Image prediction: {predicted_label} ({confidence:.3f} confidence)")
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
            "used_tta": False,
            "error": str(e)
        }

# ========== ENHANCED VIDEO INFERENCE ==========
def predict_video(video_path, max_frames=10):
    """
    Enhanced video prediction with frame sampling and temporal consistency
    
    Args:
        video_path: Path to video file
        max_frames: Maximum number of frames to analyze
    
    Returns:
        Dictionary with prediction results
    """
    try:
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Smart frame sampling
        if total_frames <= max_frames:
            frame_indices = list(range(total_frames))
        else:
            # Sample frames evenly throughout the video
            frame_indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
        
        frame_predictions = []
        processed_frames = 0
        
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                continue
                
            # Convert frame to PIL Image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            
            # Predict on frame
            tensor = inference_transform(img).unsqueeze(0).to(device)
            
            with torch.no_grad():
                logits = model(tensor)
                probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
                frame_predictions.append(probs)
                processed_frames += 1
        
        cap.release()
        
        if not frame_predictions:
            raise ValueError("No frames could be processed")
        
        # Aggregate predictions with temporal consistency
        frame_predictions = np.array(frame_predictions)
        
        # Weighted average (give more weight to middle frames)
        weights = np.exp(-0.1 * np.abs(np.arange(len(frame_predictions)) - len(frame_predictions)//2))
        weights = weights / weights.sum()
        
        weighted_probs = np.average(frame_predictions, axis=0, weights=weights)
        
        # Get final prediction
        top_idx = int(np.argmax(weighted_probs))
        confidence = float(np.max(weighted_probs))
        predicted_label = index_to_label[top_idx]
        
        # Calculate temporal consistency (how consistent predictions are across frames)
        frame_labels = [np.argmax(pred) for pred in frame_predictions]
        consistency = np.mean([label == top_idx for label in frame_labels])
        
        # Determine confidence level
        if confidence >= confidence_thresholds["high"] and consistency >= 0.7:
            confidence_level = "high"
        elif confidence >= confidence_thresholds["medium"] and consistency >= 0.5:
            confidence_level = "medium"
        else:
            confidence_level = "low"
        
        class_probs = {
            index_to_label[i]: float(p) for i, p in enumerate(weighted_probs)
        }
        
        result = {
            "label": predicted_label,
            "confidence": confidence,
            "confidence_level": confidence_level,
            "class_probs": class_probs,
            "probabilities": class_probs,
            "description": label_descriptions.get(predicted_label, predicted_label),
            "model_type": model_type,
            "frames_analyzed": processed_frames,
            "temporal_consistency": float(consistency),
            "video_analysis": True
        }
        
        logger.info(f"Video prediction: {predicted_label} ({confidence:.3f} confidence, {consistency:.3f} consistency)")
        
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
            "temporal_consistency": 0.0,
            "video_analysis": True,
            "error": str(e)
        }

# ========== MODEL INFO ==========
def get_model_info():
    """Get information about the loaded model"""
    return {
        "model_type": model_type,
        "device": str(device),
        "classes": list(index_to_label.values()),
        "confidence_thresholds": confidence_thresholds,
        "supports_tta": model_type in ["enhanced", "enhanced_with_basic_weights"],
        "supports_video": True
    }

logger.info("✓ Enhanced inference module loaded successfully")
logger.info(f"Model info: {get_model_info()}")