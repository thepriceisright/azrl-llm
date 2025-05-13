#!/usr/bin/env python
# coding: utf-8

# # Absolute Zero Reinforcement Learning
# 
# This notebook demonstrates how to use the AZRL implementation to perform a single training iteration with the Qwen2.5-Coder-3B model.

# Add parent directory to path
import os
import sys
sys.path.insert(0, os.path.abspath('..'))

# Import required modules
import torch
from src.orchestrator import AZRPipeline
from src.llm import get_model_service
from utils import get_logger, get_config

# ## Setup
# 
# First, let's create required directories for the system, load the model, and initialize the pipeline.

# Create required directories
os.makedirs("../workspace/buffer", exist_ok=True)
os.makedirs("../workspace/checkpoints", exist_ok=True)
os.makedirs("../workspace/logs", exist_ok=True)

# Set random seed for reproducibility
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
import random
import numpy as np
random.seed(42)
np.random.seed(42)

# Initialize logger
logger = get_logger("notebook")
logger.log("Initializing AZRL components...")

# Load the model (this will take some time)
model_service = get_model_service()
model = model_service.model
tokenizer = model_service.tokenizer

# Print model info
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"Model loaded: {model_service.model_name}")
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# ## Initialize the Pipeline
# 
# Now, let's initialize the AZR pipeline and run a single training iteration.

# Initialize pipeline
pipeline = AZRPipeline()
logger.log("AZR pipeline initialized")

# ## Run a Test Iteration
# 
# Let's run a single training iteration to test that everything works properly.

# Run a single iteration
logger.log("Running a test iteration...")
metrics = pipeline.run_iteration()
print(f"\nIteration completed with task type: {metrics['task_type']}")
print(f"Proposer reward: {metrics['proposer_reward']:.4f}")
print(f"Solver accuracy: {metrics['solver_accuracy']:.4f}")
print(f"Buffer sizes: {metrics['buffer_sizes']}")

# ## Next Steps
# 
# For a full training run, use the `main.py` script with the desired number of iterations:
# 
# ```
# python main.py --iterations 100
# ```
# 
# This will run the full training loop for 100 iterations, logging results and saving checkpoints periodically. 