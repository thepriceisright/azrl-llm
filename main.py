#!/usr/bin/env python
"""
Main entry point for the Absolute Zero Reinforcement Learning (AZRL) pipeline.

This script initializes and runs the AZR pipeline for the specified number of iterations.
"""
import os
import argparse
import torch
import time
import threading
import datetime
import sys
from typing import Optional, Dict, Any

from utils import get_logger, get_config
from src.orchestrator import AZRPipeline


# Global variables for heartbeat thread
running = False
last_status = "Initializing"
heartbeat_interval = 300  # 5 minutes by default


def heartbeat_thread():
    """
    Thread function that periodically outputs a heartbeat message to show the process is alive.
    """
    global running, last_status
    logger = get_logger("heartbeat")
    
    start_time = time.time()
    last_heartbeat = start_time
    
    while running:
        now = time.time()
        # Output heartbeat message every interval seconds
        if now - last_heartbeat >= heartbeat_interval:
            elapsed = now - start_time
            elapsed_hours = int(elapsed // 3600)
            elapsed_minutes = int((elapsed % 3600) // 60)
            elapsed_seconds = int(elapsed % 60)
            
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            logger.log(f"[{timestamp}] ❤️ HEARTBEAT ❤️ Process running for {elapsed_hours}h {elapsed_minutes}m {elapsed_seconds}s")
            logger.log(f"Current status: {last_status}")
            
            # Force flush stdout to ensure message is visible immediately
            sys.stdout.flush()
            
            last_heartbeat = now
        
        # Sleep to avoid excessive CPU usage
        time.sleep(10)


def update_status(status: str):
    """
    Update the global status variable that's displayed in heartbeat messages.
    
    Args:
        status: New status message
    """
    global last_status
    last_status = status


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
    parser.add_argument(
        "--heartbeat",
        type=int,
        default=300,
        help="Interval in seconds between heartbeat messages (default: 300)"
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
    seed: int = 42,
    heartbeat_seconds: int = 300
):
    """
    Main entry point for the AZRL pipeline.
    
    Args:
        config_path: Path to the configuration file
        iterations: Number of iterations to run
        checkpoint_path: Path to checkpoint to load (optional)
        seed: Random seed for reproducibility
        heartbeat_seconds: Interval between heartbeat messages
    """
    global running, heartbeat_interval
    
    # Set random seed
    set_seed(seed)
    
    # Initialize config, logger, and other services
    if config_path:
        from utils.config_utils import ConfigManager
        ConfigManager(config_path)  # Initialize global config with custom path
    
    logger = get_logger("main")
    logger.log(f"Starting AZRL with {iterations} iterations")
    
    # Configure and start heartbeat thread
    heartbeat_interval = heartbeat_seconds
    running = True
    heartbeat = threading.Thread(target=heartbeat_thread)
    heartbeat.daemon = True  # Thread will terminate when main process exits
    heartbeat.start()
    
    try:
        # Initialize pipeline
        update_status("Initializing AZR pipeline")
        pipeline = AZRPipeline()
        
        # Load checkpoint if specified
        if checkpoint_path:
            update_status(f"Loading checkpoint from {checkpoint_path}")
            logger.log(f"Loading checkpoint from {checkpoint_path}")
            from src.llm import get_model_service
            model_service = get_model_service()
            model_service.load_checkpoint(checkpoint_path)
        
        # Run training
        update_status(f"Beginning training loop with {iterations} iterations")
        metrics = pipeline.train(iterations)
        
        # Log final metrics
        update_status("Training completed")
        logger.log(f"Training completed. Final metrics: {metrics[-1] if metrics else 'No metrics'}")
        
        return metrics
    
    finally:
        # Ensure heartbeat thread is terminated
        running = False
        heartbeat.join(timeout=1.0)
        logger.log("Exiting AZRL")


if __name__ == "__main__":
    args = parse_args()
    main(
        config_path=args.config,
        iterations=args.iterations,
        checkpoint_path=args.checkpoint,
        seed=args.seed,
        heartbeat_seconds=args.heartbeat
    ) 