import os
import time
import logging
from PIL import Image
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models, datasets
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===========================================
# ENHANCED MODEL ARCHITECTURE
# ===========================================
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

# ===========================================
# CONFIGURATION
# ===========================================
# Update these paths according to your dataset location
TRAIN_DIR = r"C:/Users/anmol/OneDrive/Desktop/Hackathon/DeepDetect/Datasets/train"
VAL_DIR = r"C:/Users/anmol/OneDrive/Desktop/Hackathon/DeepDetect/Datasets/Val"
CHECKPOINT_DIR = r"C:/Users/anmol/OneDrive/Desktop/DeepDetect/checkpoints"

# Create checkpoint directory
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Training settings
BATCH_SIZE = 12  # Reduced for enhanced model
NUM_EPOCHS = 8
LEARNING_RATE = 5e-5  # Lower learning rate for fine-tuning
NUM_WORKERS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger.info(f"Using device: {DEVICE}")
logger.info(f"Training configuration:")
logger.info(f"  Batch size: {BATCH_SIZE}")
logger.info(f"  Epochs: {NUM_EPOCHS}")
logger.info(f"  Learning rate: {LEARNING_RATE}")

# ===========================================
# ENHANCED DATA AUGMENTATION
# ===========================================
train_transform = transforms.Compose([
    transforms.Resize((400, 400)),  # Slightly larger for better features
    transforms.RandomResizedCrop(380, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=8),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
    transforms.RandomGrayscale(p=0.1),
    transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 2.0))], p=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.1, scale=(0.02, 0.1))  # Random erasing for robustness
])

val_transform = transforms.Compose([
    transforms.Resize((380, 380)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ===========================================
# FOCAL LOSS FOR IMBALANCED DATA
# ===========================================
class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance"""
    
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# ===========================================
# TRAINING FUNCTION
# ===========================================
def train_enhanced_model():
    """Train the enhanced deepfake detection model"""
    
    # Load datasets
    logger.info("Loading datasets...")
    train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
    val_dataset = datasets.ImageFolder(VAL_DIR, transform=val_transform)
    
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")
    logger.info(f"Classes: {train_dataset.classes}")
    logger.info(f"Class to index: {train_dataset.class_to_idx}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True if DEVICE.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True if DEVICE.type == 'cuda' else False
    )
    
    # Initialize model
    logger.info("Initializing enhanced model...")
    model = EnhancedDeepfakeDetector(num_classes=len(train_dataset.classes), dropout_rate=0.3)
    model = model.to(DEVICE)
    
    # Loss function and optimizer
    criterion = FocalLoss(alpha=1, gamma=2)  # Use focal loss for better handling of imbalanced data
    
    # Optimizer with different learning rates for backbone and classifier
    backbone_params = []
    classifier_params = []
    
    for name, param in model.named_parameters():
        if 'backbone' in name:
            backbone_params.append(param)
        else:
            classifier_params.append(param)
    
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': LEARNING_RATE * 0.1},  # Lower LR for pretrained backbone
        {'params': classifier_params, 'lr': LEARNING_RATE}       # Higher LR for new layers
    ], weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=3, T_mult=2, eta_min=1e-7
    )
    
    # Training loop
    best_val_acc = 0.0
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    logger.info("Starting training...")
    
    for epoch in range(NUM_EPOCHS):
        epoch_start = time.time()
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]")
        
        for batch_idx, (images, labels) in enumerate(train_pbar):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            # Update progress bar
            train_pbar.set_postfix({
                'Loss': f'{train_loss/(batch_idx+1):.4f}',
                'Acc': f'{100.*train_correct/train_total:.2f}%',
                'LR': f'{optimizer.param_groups[0]["lr"]:.2e}'
            })
        
        train_acc = 100. * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_predictions = []
        all_labels = []
        
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val]")
        
        with torch.no_grad():
            for images, labels in val_pbar:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                val_pbar.set_postfix({
                    'Loss': f'{val_loss/(len(all_predictions)//BATCH_SIZE+1):.4f}',
                    'Acc': f'{100.*val_correct/val_total:.2f}%'
                })
        
        val_acc = 100. * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        
        # Update learning rate
        scheduler.step()
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start
        
        # Print epoch results
        logger.info(f"\nEpoch {epoch+1}/{NUM_EPOCHS} Results:")
        logger.info(f"  Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        logger.info(f"  Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        logger.info(f"  Time: {epoch_time:.2f}s | LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = avg_val_loss
            patience_counter = 0
            
            # Save enhanced model
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "enhanced_model.pth"))
            logger.info(f"  ✓ New best model saved! Val Acc: {val_acc:.2f}%")
            
            # Generate classification report for best model
            class_names = train_dataset.classes
            report = classification_report(all_labels, all_predictions, 
                                         target_names=class_names, digits=4)
            logger.info(f"\nClassification Report (Best Model):\n{report}")
            
        else:
            patience_counter += 1
            logger.info(f"  No improvement. Patience: {patience_counter}/{patience}")
        
        # Early stopping
        if patience_counter >= patience:
            logger.info(f"Early stopping triggered after {epoch+1} epochs")
            break
        
        # Save checkpoint every few epochs
        if (epoch + 1) % 3 == 0:
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f"enhanced_model_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(f"  Checkpoint saved: {checkpoint_path}")
    
    logger.info(f"\nTraining completed!")
    logger.info(f"Best validation accuracy: {best_val_acc:.2f}%")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    logger.info(f"Enhanced model saved to: {os.path.join(CHECKPOINT_DIR, 'enhanced_model.pth')}")

if __name__ == "__main__":
    # Check if datasets exist
    if not os.path.exists(TRAIN_DIR):
        logger.error(f"Training directory not found: {TRAIN_DIR}")
        logger.error("Please update TRAIN_DIR path in the script")
        exit(1)
    
    if not os.path.exists(VAL_DIR):
        logger.error(f"Validation directory not found: {VAL_DIR}")
        logger.error("Please update VAL_DIR path in the script")
        exit(1)
    
    # Start training
    train_enhanced_model()