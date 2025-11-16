#!/usr/bin/env python3
"""
Simple runner for the Multi-Task Gradio App
"""

import warnings
warnings.filterwarnings("ignore")

import os
import sys

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import main

if __name__ == "__main__":
    print("Starting Multi-Task Gradio App...")
    print("This may take a few minutes to load all models...")
    print("App will be available at: http://localhost:7860")
    main()