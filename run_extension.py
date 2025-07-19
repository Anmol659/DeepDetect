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

def check_your_trained_model():
    """Check if YOUR trained model checkpoint exists"""
    possible_paths = [
        Path("checkpoints/best_model_3way.pth"),
        Path("backend/checkpoints/best_model_3way.pth"),
        Path("best_model_3way.pth")
    ]
    
    for checkpoint_path in possible_paths:
        if checkpoint_path.exists():
            print(f"✓ YOUR trained model found: {checkpoint_path}")
            return True, checkpoint_path
    
    print("❌ YOUR trained model checkpoint not found!")
    print("Searched in:")
    for path in possible_paths:
        print(f"  - {path}")
    print("\nTo fix this:")
    print("1. Train your model using: cd backend/Model_A && python MDA_2.py")
    print("2. Or copy your existing best_model_3way.pth to the checkpoints/ directory")
    print("3. Or place it in the project root directory")
    return False, None

def create_checkpoints_dir():
    """Create checkpoints directory if it doesn't exist"""
    checkpoints_dir = Path("checkpoints")
    if not checkpoints_dir.exists():
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created checkpoints directory: {checkpoints_dir}")

def start_flask_server():
    """Start the Flask backend server"""
    print("Starting DeepDetect Flask server with YOUR trained model...")
    
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
        print("Current directory:", os.getcwd())
        sys.exit(1)
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    # Create checkpoints directory
    create_checkpoints_dir()
    
    # Check for YOUR trained model
    model_found, model_path = check_your_trained_model()
    if not model_found:
        print("\n❌ Cannot start without YOUR trained model!")
        print("\nOptions:")
        print("1. Train the model: cd backend/Model_A && python MDA_2.py")
        print("2. Copy your existing model file to checkpoints/best_model_3way.pth")
        
        choice = input("\nDo you want to continue anyway? (y/n): ").lower().strip()
        if choice != 'y':
            print("Exiting. Please add your trained model and try again.")
            sys.exit(1)
    else:
        print(f"✓ Using YOUR trained model from: {model_path}")
    
    # Print extension setup instructions
    print_extension_instructions()
    
    # Ask user if they want to start the server
    start_server = input("\nStart the Flask server now? (y/n): ").lower().strip()
    if start_server == 'y':
        start_flask_server()
    else:
        print("You can start the server later by running: python backend/app.py")

if __name__ == "__main__":
    main()