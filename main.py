#!/usr/bin/env python
"""
Main entry point for the Absolute Zero Reinforcement Learning (AZRL) pipeline.

This script initializes and runs the AZR pipeline for the specified number of iterations.
"""
import os
import argparse
import torch
from typing import Optional

from utils import get_logger, get_config
from src.orchestrator import AZRPipeline


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run Absolute Zero Reinforcement Learning")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to the configuration file"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of iterations to run"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to load (optional)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)


def main(
    config_path: Optional[str] = None,
    iterations: int = 100,
    checkpoint_path: Optional[str] = None,
    seed: int = 42
):
    """
    Main entry point for the AZRL pipeline.
    
    Args:
        config_path: Path to the configuration file
        iterations: Number of iterations to run
        checkpoint_path: Path to checkpoint to load (optional)
        seed: Random seed for reproducibility
    """
    # Set random seed
    set_seed(seed)
    
    # Initialize config, logger, and other services
    if config_path:
        from utils.config_utils import ConfigManager
        ConfigManager(config_path)  # Initialize global config with custom path
    
    logger = get_logger("main")
    logger.log(f"Starting AZRL with {iterations} iterations")
    
    # Initialize pipeline
    pipeline = AZRPipeline()
    
    # Load checkpoint if specified
    if checkpoint_path:
        logger.log(f"Loading checkpoint from {checkpoint_path}")
        from src.llm import get_model_service
        model_service = get_model_service()
        model_service.load_checkpoint(checkpoint_path)
    
    # Run training
    metrics = pipeline.train(iterations)
    
    # Log final metrics
    logger.log(f"Training completed. Final metrics: {metrics[-1] if metrics else 'No metrics'}")
    
    return metrics


if __name__ == "__main__":
    args = parse_args()
    main(
        config_path=args.config,
        iterations=args.iterations,
        checkpoint_path=args.checkpoint,
        seed=args.seed
    ) 