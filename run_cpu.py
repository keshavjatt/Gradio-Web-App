#!/usr/bin/env python3
"""
CPU-Only runner for the Multi-Task Gradio App
"""

import warnings
warnings.filterwarnings("ignore")

import os
import sys

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Set environment variables to disable CUDA and xFormers
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["XFORMERS_DISABLED"] = "1"

print("🚀 Starting Multi-Task Gradio App (CPU-Only Mode)...")
print("📝 This may take a few minutes to load all models...")
print("🌐 App will be available at: http://localhost:7860")
print("⚡ Running in CPU-Only mode for maximum compatibility")

try:
    from app import main
    main()
except Exception as e:
    print(f"❌ Error starting app: {e}")
    print("🔧 Try installing dependencies: pip install -r requirements.txt")