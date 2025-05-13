#!/bin/bash

# Create directories for persistent storage
mkdir -p /workspace/buffer
mkdir -p /workspace/checkpoints
mkdir -p /workspace/logs

# Ensure the configuration has the correct paths
if [ ! -d "/workspace/azrl-llm" ]; then
    echo "ERROR: This script should be run from the RunPod workspace with your repository cloned to /workspace/azrl-llm"
    exit 1
fi

# Display configuration information
echo "Workspace initialization complete."
echo "To run training with the mock executor (no Docker required):"
echo "cd /workspace/azrl-llm"
echo "WANDB_MODE=disabled python main.py --iterations 100"
echo ""
echo "Make sure executor.use_mock is set to true in config/config.yaml"
echo "Current configuration:"
grep "use_mock" /workspace/azrl-llm/config/config.yaml || echo "use_mock not found in config, please add it manually." 