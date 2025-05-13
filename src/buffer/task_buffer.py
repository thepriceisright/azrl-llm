import os
import json
import random
import jsonlines
from typing import Dict, List, Tuple, Any, Optional, Union
import threading
from collections import defaultdict

from utils import get_logger, get_config


class TaskBuffer:
    """
    Buffer for storing and sampling tasks for the Absolute Zero reinforcement learning.
    
    The buffer stores three types of tasks:
    - Abduction: Given program p and output o, find input i such that p(i) = o
    - Deduction: Given program p and input i, find output o such that p(i) = o
    - Induction: Given a subset of input-output pairs (i,o), find program p such that p(i) = o for all pairs
    """
    def __init__(self, buffer_dir: Optional[str] = None):
        """
        Initialize the task buffer.
        
        Args:
            buffer_dir: Directory to store the buffer files. If None, uses the config value.
        """
        self.logger = get_logger("task_buffer")
        self.config = get_config()
        
        # Get configuration
        self.buffer_dir = buffer_dir or self.config.get("paths.buffer_dir")
        self.max_buffer_size = self.config.get("training.max_buffer_size", 16384)
        
        # Create buffer directory if it doesn't exist
        os.makedirs(self.buffer_dir, exist_ok=True)
        
        # Initialize buffers for each task type
        self.buffers = {
            "abduction": [],  # (program, input, output) triplets
            "deduction": [],  # (program, input, output) triplets
            "induction": []   # (program, [(input, output)], metaprompt) entries
        }
        
        # Locks for thread safety
        self.locks = {
            "abduction": threading.Lock(),
            "deduction": threading.Lock(),
            "induction": threading.Lock()
        }
        
        # Load existing buffers if they exist
        self._load_buffers()
    
    def _get_buffer_path(self, task_type: str) -> str:
        """
        Get the path to the buffer file for a task type.
        
        Args:
            task_type: Type of task (abduction, deduction, induction)
            
        Returns:
            Path to the buffer file
        """
        return os.path.join(self.buffer_dir, f"{task_type}_buffer.jsonl")
    
    def _load_buffers(self) -> None:
        """Load all buffers from disk if they exist."""
        for task_type in self.buffers:
            buffer_path = self._get_buffer_path(task_type)
            
            if os.path.exists(buffer_path):
                with self.locks[task_type]:
                    self.buffers[task_type] = []
                    try:
                        with jsonlines.open(buffer_path) as reader:
                            for item in reader:
                                self.buffers[task_type].append(item)
                        
                        self.logger.log(f"Loaded {len(self.buffers[task_type])} items for {task_type} from {buffer_path}")
                    except Exception as e:
                        self.logger.log(f"Error loading {task_type} buffer: {e}", level="ERROR")
    
    def _save_buffer(self, task_type: str) -> None:
        """
        Save a buffer to disk.
        
        Args:
            task_type: Type of task (abduction, deduction, induction)
        """
        buffer_path = self._get_buffer_path(task_type)
        
        with self.locks[task_type]:
            try:
                with jsonlines.open(buffer_path, mode='w') as writer:
                    for item in self.buffers[task_type]:
                        writer.write(item)
                
                self.logger.log(f"Saved {len(self.buffers[task_type])} items for {task_type} to {buffer_path}")
            except Exception as e:
                self.logger.log(f"Error saving {task_type} buffer: {e}", level="ERROR")
    
    def add_task(self, task_type: str, task_data: Dict[str, Any]) -> None:
        """
        Add a task to the buffer.
        
        Args:
            task_type: Type of task (abduction, deduction, induction)
            task_data: Task data to add
        """
        if task_type not in self.buffers:
            raise ValueError(f"Unknown task type: {task_type}")
        
        with self.locks[task_type]:
            # Add to buffer
            self.buffers[task_type].append(task_data)
            
            # Trim buffer if it exceeds max size
            if len(self.buffers[task_type]) > self.max_buffer_size:
                # Remove random items to get down to the max size
                # This is a simple strategy; could be replaced with more sophisticated eviction policies
                excess = len(self.buffers[task_type]) - self.max_buffer_size
                to_remove = random.sample(range(len(self.buffers[task_type])), excess)
                to_remove.sort(reverse=True)  # Remove from end to avoid index issues
                
                for idx in to_remove:
                    self.buffers[task_type].pop(idx)
            
            # Save buffer to disk
            self._save_buffer(task_type)
    
    def sample_tasks(self, task_type: str, k: int) -> List[Dict[str, Any]]:
        """
        Sample k tasks from the buffer for a given task type.
        
        Args:
            task_type: Type of task (abduction, deduction, induction)
            k: Number of tasks to sample
            
        Returns:
            List of sampled tasks
        """
        if task_type not in self.buffers:
            raise ValueError(f"Unknown task type: {task_type}")
        
        with self.locks[task_type]:
            buffer = self.buffers[task_type]
            
            if not buffer:
                return []
            
            # Sample k tasks, with replacement if k > len(buffer)
            if k <= len(buffer):
                return random.sample(buffer, k)
            else:
                # With replacement if k > len(buffer)
                return [random.choice(buffer) for _ in range(k)]
    
    def get_buffer_size(self, task_type: Optional[str] = None) -> Union[int, Dict[str, int]]:
        """
        Get the size of a buffer or all buffers.
        
        Args:
            task_type: Optional task type. If None, returns sizes for all buffers.
            
        Returns:
            Buffer size or dictionary of buffer sizes
        """
        if task_type is not None:
            if task_type not in self.buffers:
                raise ValueError(f"Unknown task type: {task_type}")
            
            with self.locks[task_type]:
                return len(self.buffers[task_type])
        else:
            # Return sizes for all buffers
            sizes = {}
            for task_type in self.buffers:
                with self.locks[task_type]:
                    sizes[task_type] = len(self.buffers[task_type])
            return sizes
    
    def initialize_with_seed_examples(self) -> None:
        """
        Initialize the buffer with seed examples if empty.
        
        For AZR, this typically includes the identity function triplet,
        but can be expanded to include other basic examples.
        """
        # Check if buffers are empty
        buffer_sizes = self.get_buffer_size()
        if sum(buffer_sizes.values()) > 0:
            self.logger.log("Buffers already contain data, skipping seed initialization")
            return
        
        # Add identity function as the initial seed
        identity_program = "def solution(x):\n    return x"
        
        # Sample inputs and generate outputs for the identity function
        example_inputs = [42, "hello", [1, 2, 3], {"a": 1, "b": 2}]
        
        # Add to all three buffer types
        for input_val in example_inputs:
            # Identity function, so output equals input
            output_val = input_val
            
            # Add to deduction buffer
            self.add_task("deduction", {
                "program": identity_program,
                "input": input_val,
                "output": output_val
            })
            
            # Add to abduction buffer
            self.add_task("abduction", {
                "program": identity_program,
                "input": input_val,
                "output": output_val
            })
        
        # Add to induction buffer (need multiple input-output pairs)
        input_output_pairs = [(x, x) for x in example_inputs]
        metaprompt = "Write a function that returns the input unchanged."
        
        self.add_task("induction", {
            "program": identity_program,
            "examples": input_output_pairs,
            "metaprompt": metaprompt
        })
        
        self.logger.log(f"Initialized buffers with identity function examples")
        self.logger.log(f"Buffer sizes: {self.get_buffer_size()}")


# Singleton instance
_task_buffer = None


def get_task_buffer() -> TaskBuffer:
    """
    Get or create the singleton task buffer instance.
    
    Returns:
        TaskBuffer instance
    """
    global _task_buffer
    if _task_buffer is None:
        _task_buffer = TaskBuffer()
    return _task_buffer 