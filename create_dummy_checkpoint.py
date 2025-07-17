#!/usr/bin/env python3
"""
Create a dummy model checkpoint for testing purposes
This allows the server to run even without a trained model
"""

import torch
import torch.nn as nn
from torchvision import models
import os

def create_dummy_checkpoint():
    """Create a dummy model checkpoint with random weights"""
    print("Creating dummy model checkpoint for testing...")
    
    # Create checkpoints directory if it doesn't exist
    os.makedirs("checkpoints", exist_ok=True)
    
    # Create the model architecture
    weights = models.EfficientNet_B4_Weights.IMAGENET1K_V1
    model = models.efficientnet_b4(weights=weights)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 3)
    
    # Initialize the classifier with random weights
    nn.init.xavier_uniform_(model.classifier[1].weight)
    nn.init.zeros_(model.classifier[1].bias)
    
    # Save the model state dict
    checkpoint_path = "checkpoints/best_model_3way.pth"
    torch.save(model.state_dict(), checkpoint_path)
    
    print(f"✓ Dummy checkpoint created at: {checkpoint_path}")
    print("⚠ Note: This is a dummy model with random weights.")
    print("  For accurate results, train the model using the scripts in backend/Model_A/")
    
    return checkpoint_path

if __name__ == "__main__":
    create_dummy_checkpoint()