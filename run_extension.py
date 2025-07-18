#!/usr/bin/env python3
"""
Script to run the DeepDetect Flask backend for Chrome extension integration
"""

import os
import sys
import subprocess
import webbrowser
import time
from pathlib import Path

def check_requirements():
    """Check if required packages are installed"""
    try:
        import flask
        import torch
        import torchvision
        import cv2
        import PIL
        print("✓ All required packages are installed")
        return True
    except ImportError as e:
        print(f"✗ Missing required package: {e}")
        print("Please install requirements: pip install -r requirements.txt")
        return False

def check_model_files():
    """Check if model checkpoint exists"""
    checkpoint_path = Path("checkpoints/best_model_3way.pth")
    if not checkpoint_path.exists():
        print(f"⚠ Model checkpoint not found: {checkpoint_path}")
        print("The server will run in fallback mode with limited accuracy.")
        print("For full functionality, please:")
        print("  1. Train the model using the training scripts in backend/Model_A/")
        print("  2. Copy the resulting 'best_model_3way.pth' to the checkpoints/ directory")
        print("  3. Or download a pre-trained model if available")
        return False
    print("✓ Model checkpoint found")
    return True

def start_flask_server():
    """Start the Flask backend server"""
    print("Starting DeepDetect Flask server...")
    
    # Change to backend directory
    backend_dir = Path("backend")
    if backend_dir.exists():
        os.chdir(backend_dir)
    
    # Set environment variables
    os.environ['FLASK_APP'] = 'app.py'
    os.environ['FLASK_ENV'] = 'development'
    
    try:
        # Start Flask server
        subprocess.run([sys.executable, 'app.py'], check=True)
    except KeyboardInterrupt:
        print("\n✓ Server stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"✗ Server failed to start: {e}")
        return False
    
    return True

def print_extension_instructions():
    """Print instructions for loading the Chrome extension"""
    print("\n" + "="*60)
    print("CHROME EXTENSION SETUP INSTRUCTIONS")
    print("="*60)
    print("1. Open Chrome and go to: chrome://extensions/")
    print("2. Enable 'Developer mode' (toggle in top right)")
    print("3. Click 'Load unpacked'")
    print("4. Select the 'extension' folder from this project")
    print("5. The DeepDetect extension should now appear in your extensions")
    print("6. Pin the extension to your toolbar for easy access")
    print("\nThe extension will connect to: http://localhost:5000")
    print("Make sure the Flask server is running before using the extension!")
    print("="*60)

def main():
    print("DeepDetect Chrome Extension Setup")
    print("="*40)
    
    # Check if we're in the right directory
    if not Path("backend/app.py").exists():
        print("✗ Please run this script from the project root directory")
        sys.exit(1)
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    # Check model files
    model_available = check_model_files()
    if not model_available:
        print("\n⚠ Continuing without trained model (fallback mode)")
    
    # Print extension setup instructions
    print_extension_instructions()
    
    # Start the server automatically
    start_flask_server()

if __name__ == "__main__":
    main()