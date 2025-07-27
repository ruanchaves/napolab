#!/usr/bin/env python3
"""
Napolab Leaderboard Launcher Script

This script checks dependencies and launches the Gradio app for the Napolab leaderboard.
"""

import sys
import subprocess
import importlib.util
from pathlib import Path

def check_dependency(package_name):
    """Check if a package is installed."""
    spec = importlib.util.find_spec(package_name)
    return spec is not None

def install_dependencies():
    """Install required dependencies."""
    print("Installing required dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def main():
    """Main launcher function."""
    print("🚀 Napolab Leaderboard Launcher")
    print("=" * 40)
    
    # Check if we're in the right directory
    if not Path("app.py").exists():
        print("❌ Error: app.py not found. Please run this script from the leaderboard directory.")
        sys.exit(1)
    
    # Check required dependencies
    required_packages = ["gradio", "pandas", "numpy", "datasets", "plotly"]
    missing_packages = []
    
    for package in required_packages:
        if not check_dependency(package):
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing dependencies: {', '.join(missing_packages)}")
        print("Installing dependencies...")
        if not install_dependencies():
            print("❌ Failed to install dependencies. Please install them manually:")
            print("pip install -r requirements.txt")
            sys.exit(1)
    else:
        print("✅ All dependencies are installed!")
    
    # Launch the app
    print("\n🌐 Launching Napolab Leaderboard...")
    print("The app will be available at: http://localhost:7860")
    print("Press Ctrl+C to stop the server")
    print("-" * 40)
    
    try:
        import app
        # The app will be launched by the import
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Error launching app: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 