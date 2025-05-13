import os
import logging
import json
from datetime import datetime
from typing import Dict, Any, Optional, Union

# Optional wandb import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from utils.config_utils import get_config


class Logger:
    """
    Centralized logging utility for the AZ training pipeline.
    Handles both file logging and optional wandb integration.
    """
    def __init__(self, 
                 name: str, 
                 log_dir: Optional[str] = None,
                 log_level: str = "INFO",
                 use_wandb: Optional[bool] = None,
                 wandb_project: Optional[str] = None):
        """
        Initialize the logger.
        
        Args:
            name: Logger name (usually module/component name)
            log_dir: Directory to store log files
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            use_wandb: Whether to use Weights & Biases logging
            wandb_project: WandB project name (if use_wandb is True)
        """
        self.logger = logging.getLogger(name)
        
        # Set log level
        level = getattr(logging, log_level.upper(), logging.INFO)
        self.logger.setLevel(level)
        
        # Only add handlers if they don't exist to prevent duplicates
        if not self.logger.handlers:
            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(level)
            console_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)
            
            # File handler (if log_dir is provided or in config)
            config = get_config()
            log_dir = log_dir or config.get("logging.log_dir")
            
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")
                
                file_handler = logging.FileHandler(log_file)
                file_handler.setLevel(level)
                file_formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
                file_handler.setFormatter(file_formatter)
                self.logger.addHandler(file_handler)
        
        # WandB setup
        self.use_wandb = use_wandb if use_wandb is not None else config.get("logging.use_wandb", False)
        self.wandb_initialized = False
        
        if self.use_wandb and WANDB_AVAILABLE:
            self.wandb_project = wandb_project or config.get("logging.wandb_project")
            if not wandb.run:
                wandb.init(project=self.wandb_project, config=config.config)
                self.wandb_initialized = True
    
    def log(self, message: str, level: str = "INFO") -> None:
        """
        Log a message at the specified level.
        
        Args:
            message: The message to log
            level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        log_func = getattr(self.logger, level.lower(), self.logger.info)
        log_func(message)
    
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """
        Log metrics to WandB (if enabled) and as INFO level logs.
        
        Args:
            metrics: Dictionary of metric names and values
            step: Optional step/iteration number for WandB
        """
        # Log to file/console
        self.logger.info(f"Metrics: {json.dumps(metrics)}")
        
        # Log to WandB if available
        if self.use_wandb and WANDB_AVAILABLE and self.wandb_initialized:
            wandb.log(metrics, step=step)
    
    def log_artifact(self, name: str, artifact_type: str, path: str) -> None:
        """
        Log an artifact to WandB (if enabled).
        
        Args:
            name: Name of the artifact
            artifact_type: Type of artifact (e.g., "model", "dataset")
            path: Path to the artifact file or directory
        """
        if self.use_wandb and WANDB_AVAILABLE and self.wandb_initialized:
            artifact = wandb.Artifact(name=name, type=artifact_type)
            artifact.add_file(path) if os.path.isfile(path) else artifact.add_dir(path)
            wandb.log_artifact(artifact)


# Cache for loggers to avoid creating duplicates
_loggers = {}


def get_logger(name: str) -> Logger:
    """
    Get or create a logger with the specified name.
    
    Args:
        name: Logger name (usually module/component name)
        
    Returns:
        Logger instance
    """
    if name not in _loggers:
        _loggers[name] = Logger(name)
    return _loggers[name] 