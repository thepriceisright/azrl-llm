import os
import torch
from typing import Dict, List, Tuple, Any, Optional, Union
import time

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoConfig
)
from peft import get_peft_model, LoraConfig, TaskType

from utils import get_logger, get_config


class ModelService:
    """
    Service for loading and interacting with the Qwen2.5-Coder-3B model.
    """
    def __init__(self, 
                model_name: Optional[str] = None, 
                use_lora: bool = True,
                load_in_8bit: bool = True,
                load_in_4bit: bool = False,
                device: Optional[str] = None):
        """
        Initialize the model service.
        
        Args:
            model_name: Name of the model to load. If None, uses the config value.
            use_lora: Whether to use LoRA for parameter-efficient fine-tuning.
            load_in_8bit: Whether to load the model in 8-bit precision.
            load_in_4bit: Whether to load the model in 4-bit precision.
            device: Device to load the model on. If None, uses CUDA if available, otherwise CPU.
        """
        self.logger = get_logger("model_service")
        self.config = get_config()
        
        # Get configuration
        self.model_name = model_name or self.config.get("model.name")
        self.max_prompt_length = self.config.get("model.max_prompt_length", 6144)
        self.max_response_length = self.config.get("model.max_response_length", 8096)
        
        # Device setup
        if device:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.use_lora = use_lora
        
        # Initialize model and tokenizer
        self.logger.log(f"Loading model {self.model_name}...")
        
        start_time = time.time()
        
        # Quantization config
        if load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
            self.logger.log("Using 4-bit quantization")
        elif load_in_8bit:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True
            )
            self.logger.log("Using 8-bit quantization")
        else:
            quantization_config = None
            self.logger.log("Using full precision")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Load model
        if quantization_config:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.bfloat16
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="auto" if self.device == "cuda" else None,
                torch_dtype=torch.bfloat16
            )
            if self.device == "cpu":
                self.model = self.model.to(self.device)
        
        # Add LoRA adapter if needed
        if self.use_lora:
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            )
            
            # Only wrap in PEFT if not already wrapped
            if not getattr(self.model, "is_peft_model", False):
                self.model = get_peft_model(self.model, peft_config)
                self.logger.log("Added LoRA adapter to the model")
        
        # Set generation config
        self.model.config.pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id else self.tokenizer.eos_token_id
        
        load_time = time.time() - start_time
        self.logger.log(f"Model loaded in {load_time:.2f} seconds")
        
        # Report model parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        
        self.logger.log(f"Model has {total_params:,} total parameters, {trainable_params:,} trainable")
        
        # Special tokens
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def generate(self, 
                prompt: str,
                temperature: float = 1.0,
                top_p: float = 1.0,
                max_tokens: Optional[int] = None,
                do_sample: bool = True,
                num_return_sequences: int = 1) -> Union[str, List[str]]:
        """
        Generate text based on a prompt.
        
        Args:
            prompt: The prompt for text generation
            temperature: Sampling temperature (higher = more random)
            top_p: Nucleus sampling parameter (1.0 = no filtering)
            max_tokens: Maximum number of tokens to generate. If None, uses config default.
            do_sample: Whether to use sampling (True) or greedy decoding (False)
            num_return_sequences: Number of sequences to return
            
        Returns:
            Generated text or list of generated texts
        """
        if max_tokens is None:
            max_tokens = self.max_response_length
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                num_return_sequences=num_return_sequences,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        # Decode outputs
        results = []
        for i in range(num_return_sequences):
            output_text = self.tokenizer.decode(outputs[i], skip_special_tokens=True)
            
            # Remove the prompt from the output
            if output_text.startswith(prompt):
                output_text = output_text[len(prompt):]
            
            results.append(output_text)
        
        # Return a single string if only one sequence is requested
        if num_return_sequences == 1:
            return results[0]
        
        return results
    
    def save_checkpoint(self, checkpoint_dir: Optional[str] = None) -> str:
        """
        Save model checkpoint.
        
        Args:
            checkpoint_dir: Directory to save the checkpoint. If None, uses the config value.
            
        Returns:
            Path to the saved checkpoint
        """
        checkpoint_dir = checkpoint_dir or self.config.get("paths.checkpoint_dir")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        checkpoint_path = os.path.join(checkpoint_dir, f"azrl-model-{timestamp}")
        
        # If using LoRA, only save the adapter
        if self.use_lora and getattr(self.model, "is_peft_model", False):
            self.model.save_pretrained(checkpoint_path)
            self.logger.log(f"Saved LoRA adapter to {checkpoint_path}")
        else:
            # Save full model
            self.model.save_pretrained(checkpoint_path)
            self.tokenizer.save_pretrained(checkpoint_path)
            self.logger.log(f"Saved full model to {checkpoint_path}")
        
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to the checkpoint to load
        """
        # If using LoRA, load the adapter
        if self.use_lora and getattr(self.model, "is_peft_model", False):
            self.model.load_adapter(checkpoint_path)
            self.logger.log(f"Loaded LoRA adapter from {checkpoint_path}")
        else:
            # Load full model if available
            if os.path.exists(os.path.join(checkpoint_path, "config.json")):
                self.model = AutoModelForCausalLM.from_pretrained(
                    checkpoint_path,
                    device_map="auto" if self.device == "cuda" else None,
                    torch_dtype=torch.bfloat16
                )
                if self.device == "cpu":
                    self.model = self.model.to(self.device)
                
                self.logger.log(f"Loaded full model from {checkpoint_path}")
            else:
                self.logger.log(f"No full model found at {checkpoint_path}", level="WARNING")


# Singleton instance
_model_service = None


def get_model_service() -> ModelService:
    """
    Get or create the singleton model service instance.
    
    Returns:
        ModelService instance
    """
    global _model_service
    if _model_service is None:
        _model_service = ModelService()
    return _model_service 