#!/bin/bash
# Setup script for RunPod environment

set -e  # Exit on error

echo "Setting up AZRL environment on RunPod..."

# Create workspace directories
echo "Creating workspace directories..."
mkdir -p /workspace/buffer
mkdir -p /workspace/checkpoints
mkdir -p /workspace/logs

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Setup network volume links
echo "Setting up volume links..."
if [ -d "/runpod-volume" ]; then
    echo "RunPod volume found, creating symlinks..."
    
    # Create subdirectories in volume if they don't exist
    mkdir -p /runpod-volume/azrl/buffer
    mkdir -p /runpod-volume/azrl/checkpoints
    mkdir -p /runpod-volume/azrl/logs
    
    # Create symlinks to the network volume
    ln -sfn /runpod-volume/azrl/buffer /workspace/buffer
    ln -sfn /runpod-volume/azrl/checkpoints /workspace/checkpoints
    ln -sfn /runpod-volume/azrl/logs /workspace/logs
    
    echo "Symlinks created."
else
    echo "RunPod volume not found, using local directories."
fi

# Update config with workspace paths
echo "Updating config with workspace paths..."
sed -i 's|/workspace/buffer|'"$PWD"'/workspace/buffer|g' config/config.yaml
sed -i 's|/workspace/checkpoints|'"$PWD"'/workspace/checkpoints|g' config/config.yaml
sed -i 's|/workspace/logs|'"$PWD"'/workspace/logs|g' config/config.yaml

# Make main script executable
chmod +x main.py

echo "Setup complete. Run training with:"
echo "python main.py --iterations 100" 