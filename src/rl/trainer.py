import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any, Optional, Union
import numpy as np
import time
from dataclasses import dataclass
from trl import PPOTrainer, PPOConfig
from transformers import AutoTokenizer

from utils import get_logger, get_config
from src.llm.model_service import get_model_service


@dataclass
class RLExample:
    """A single example for RL training."""
    prompt: str
    completion: str
    reward: float
    role: str  # "proposer" or "solver"
    task_type: str  # "abduction", "deduction", or "induction"


class TRR:
    """
    Task-Relative REINFORCE (TRR) algorithm for the AZR training loop.
    
    This implements the TRR++ algorithm from the AZR paper (Section 3.3.5).
    """
    def __init__(self, tokenizer):
        """
        Initialize the TRR trainer.
        
        Args:
            tokenizer: Tokenizer for the model
        """
        self.logger = get_logger("trr_trainer")
        self.config = get_config()
        
        self.tokenizer = tokenizer
        
        # Get configuration
        self.learning_rate = self.config.get("training.learning_rate", 1e-6)
        self.entropy_coeff = self.config.get("training.entropy_coeff", 0.001)
    
    def compute_logprobs(self, model, input_ids, attention_mask, labels):
        """
        Compute log probabilities for the generated completions.
        
        Args:
            model: The language model
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Target token IDs
            
        Returns:
            Log probabilities tensor
        """
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                output_hidden_states=True,
                return_dict=True
            )
            
            logits = outputs.logits
            
            # Shift logits and labels for next token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Calculate log probabilities
            probs = F.softmax(shift_logits, dim=-1)
            log_probs = F.log_softmax(shift_logits, dim=-1)
            
            # Get the log probabilities of the chosen tokens
            # Gather the log probs at the indices of the labels
            gathered_log_probs = log_probs.gather(
                dim=-1, 
                index=shift_labels.unsqueeze(-1)
            ).squeeze(-1)
            
            # Create a mask for non-padding tokens
            non_pad_mask = (shift_labels != self.tokenizer.pad_token_id).float()
            
            # Apply the mask to the log probs
            masked_log_probs = gathered_log_probs * non_pad_mask
            
            # Calculate entropy
            entropy = -torch.sum(probs * log_probs, dim=-1)
            masked_entropy = entropy * non_pad_mask
            
            return masked_log_probs, masked_entropy, non_pad_mask
    
    def update(self, model, examples: List[RLExample]) -> Dict[str, float]:
        """
        Update the model weights using the TRR++ algorithm.
        
        Args:
            model: The language model to update
            examples: List of RL examples
            
        Returns:
            Dictionary of training statistics
        """
        if not examples:
            return {
                "loss": 0.0,
                "proposer_loss": 0.0,
                "solver_loss": 0.0,
                "entropy": 0.0,
                "examples_count": 0
            }
        
        self.logger.log(f"Updating model with {len(examples)} examples")
        
        # Enable training mode
        model.train()
        
        # Initialize optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate)
        
        # Process all examples in a single batch
        device = next(model.parameters()).device
        
        # Tokenize all examples
        all_input_ids = []
        all_attention_masks = []
        all_labels = []
        all_rewards = []
        all_roles = []
        
        for example in examples:
            # Tokenize prompt
            prompt_tokens = self.tokenizer(
                example.prompt,
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            
            # Tokenize completion
            completion_tokens = self.tokenizer(
                example.completion,
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            
            # Create input_ids: prompt followed by completion
            prompt_len = prompt_tokens.input_ids.shape[1]
            
            # Input is the prompt
            input_ids = prompt_tokens.input_ids
            
            # Labels: -100 for prompt, completion token IDs for completion
            labels = torch.cat([
                torch.full_like(prompt_tokens.input_ids, -100),
                completion_tokens.input_ids
            ], dim=1)
            
            # Attention mask for both prompt and completion
            attention_mask = torch.cat([
                prompt_tokens.attention_mask,
                completion_tokens.attention_mask
            ], dim=1)
            
            all_input_ids.append(input_ids)
            all_attention_masks.append(attention_mask)
            all_labels.append(labels)
            all_rewards.append(example.reward)
            all_roles.append(example.role)
        
        # Pad to max length in batch
        max_len = max(ids.shape[1] for ids in all_input_ids)
        max_label_len = max(labels.shape[1] for labels in all_labels)
        
        padded_input_ids = []
        padded_attention_masks = []
        padded_labels = []
        
        for i in range(len(all_input_ids)):
            # Pad input_ids
            padding_len = max_len - all_input_ids[i].shape[1]
            padded_ids = F.pad(
                all_input_ids[i],
                (0, padding_len),
                value=self.tokenizer.pad_token_id
            )
            padded_input_ids.append(padded_ids)
            
            # Pad attention_mask
            padded_mask = F.pad(
                all_attention_masks[i],
                (0, padding_len),
                value=0
            )
            padded_attention_masks.append(padded_mask)
            
            # Pad labels
            label_padding_len = max_label_len - all_labels[i].shape[1]
            padded_label = F.pad(
                all_labels[i],
                (0, label_padding_len),
                value=-100
            )
            padded_labels.append(padded_label)
        
        # Stack tensors
        input_ids = torch.cat(padded_input_ids).to(device)
        attention_mask = torch.cat(padded_attention_masks).to(device)
        labels = torch.cat(padded_labels).to(device)
        rewards = torch.tensor(all_rewards).to(device)
        
        # Compute log probabilities for the generated completions
        log_probs, entropy, non_pad_mask = self.compute_logprobs(
            model, input_ids, attention_mask, labels
        )
        
        # Split examples by role
        proposer_indices = [i for i, role in enumerate(all_roles) if role == "proposer"]
        solver_indices = [i for i, role in enumerate(all_roles) if role == "solver"]
        
        proposer_rewards = rewards[proposer_indices] if proposer_indices else torch.tensor([])
        solver_rewards = rewards[solver_indices] if solver_indices else torch.tensor([])
        
        # Separate losses for proposer and solver
        proposer_loss = torch.tensor(0.0).to(device)
        solver_loss = torch.tensor(0.0).to(device)
        
        # Compute REINFORCE loss for all examples together
        token_count = non_pad_mask.sum()
        mean_entropy = (entropy * non_pad_mask).sum() / token_count if token_count > 0 else torch.tensor(0.0).to(device)
        
        # Compute REINFORCE loss for proposer
        if proposer_indices:
            proposer_log_probs = log_probs[proposer_indices]
            proposer_non_pad = non_pad_mask[proposer_indices]
            proposer_token_count = proposer_non_pad.sum()
            
            if proposer_token_count > 0:
                # Sum log probs per example
                proposer_example_logprobs = []
                
                for i in range(len(proposer_indices)):
                    example_token_count = proposer_non_pad[i].sum()
                    if example_token_count > 0:
                        example_logprob = proposer_log_probs[i].sum() / example_token_count
                    else:
                        example_logprob = torch.tensor(0.0).to(device)
                    proposer_example_logprobs.append(example_logprob)
                
                proposer_example_logprobs = torch.stack(proposer_example_logprobs)
                
                # Compute REINFORCE loss for proposer
                proposer_loss = -(proposer_example_logprobs * proposer_rewards).mean()
        
        # Compute REINFORCE loss for solver
        if solver_indices:
            solver_log_probs = log_probs[solver_indices]
            solver_non_pad = non_pad_mask[solver_indices]
            solver_token_count = solver_non_pad.sum()
            
            if solver_token_count > 0:
                # Sum log probs per example
                solver_example_logprobs = []
                
                for i in range(len(solver_indices)):
                    example_token_count = solver_non_pad[i].sum()
                    if example_token_count > 0:
                        example_logprob = solver_log_probs[i].sum() / example_token_count
                    else:
                        example_logprob = torch.tensor(0.0).to(device)
                    solver_example_logprobs.append(example_logprob)
                
                solver_example_logprobs = torch.stack(solver_example_logprobs)
                
                # Compute REINFORCE loss for solver
                solver_loss = -(solver_example_logprobs * solver_rewards).mean()
        
        # Total loss
        loss = proposer_loss + solver_loss - self.entropy_coeff * mean_entropy
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Return training statistics
        stats = {
            "loss": loss.item(),
            "proposer_loss": proposer_loss.item(),
            "solver_loss": solver_loss.item(),
            "entropy": mean_entropy.item(),
            "examples_count": len(examples),
            "proposer_count": len(proposer_indices),
            "solver_count": len(solver_indices),
            "proposer_mean_reward": proposer_rewards.mean().item() if len(proposer_indices) > 0 else 0.0,
            "solver_mean_reward": solver_rewards.mean().item() if len(solver_indices) > 0 else 0.0,
        }
        
        self.logger.log(f"Training stats: {stats}")
        
        return stats


class PPOTrainingWrapper:
    """
    Alternative PPO-based training implementation.
    
    This implements PPO using the TRL library as a fallback if TRR++ has issues.
    """
    def __init__(self):
        """
        Initialize the PPO trainer wrapper.
        """
        self.logger = get_logger("ppo_trainer")
        self.config = get_config()
        
        # Get configuration
        self.learning_rate = self.config.get("training.learning_rate", 1e-6)
        self.batch_size = self.config.get("training.batch_size", 32)
        self.ppo_epochs = self.config.get("training.ppo_epochs", 1)
        self.use_kl = self.config.get("training.use_kl", False)
        
        # Get model service
        model_service = get_model_service()
        self.model = model_service.model
        self.tokenizer = model_service.tokenizer
        
        # Create PPO config
        ppo_config = PPOConfig(
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            mini_batch_size=self.batch_size,
            ppo_epochs=self.ppo_epochs,
            optimize_device_cache=True,
            use_score_scaling=False,
            use_score_norm=False,
            kl_penalty=self.use_kl,
            kl_loss_coef=0.1 if self.use_kl else 0.0,
            init_kl_coef=0.2 if self.use_kl else 0.0,
            log_with=None,
        )
        
        # Create PPO trainer
        self.ppo_trainer = PPOTrainer(
            config=ppo_config,
            model=self.model,
            tokenizer=self.tokenizer,
        )
    
    def update(self, examples: List[RLExample]) -> Dict[str, float]:
        """
        Update the model weights using the PPO algorithm.
        
        Args:
            examples: List of RL examples
            
        Returns:
            Dictionary of training statistics
        """
        if not examples:
            return {
                "loss": 0.0,
                "examples_count": 0
            }
        
        self.logger.log(f"Updating model with PPO using {len(examples)} examples")
        
        # Prepare queries and responses
        queries = [example.prompt for example in examples]
        responses = [example.completion for example in examples]
        rewards = [example.reward for example in examples]
        
        # Tokenize the queries and responses
        query_tensors = [
            self.tokenizer(query, return_tensors="pt").input_ids.squeeze(0)
            for query in queries
        ]
        response_tensors = [
            self.tokenizer(response, return_tensors="pt").input_ids.squeeze(0)
            for response in responses
        ]
        
        # Run PPO step
        stats = self.ppo_trainer.step(query_tensors, response_tensors, rewards)
        
        # Convert stats to dictionary
        stats_dict = {
            "loss": float(stats["ppo/loss/total"]),
            "policy_loss": float(stats["ppo/loss/policy"]),
            "value_loss": float(stats["ppo/loss/value"]),
            "entropy": float(stats["ppo/policy/entropy"]),
            "kl": float(stats["ppo/policy/kl"]) if self.use_kl else 0.0,
            "mean_reward": float(stats["ppo/rewards/mean"]),
            "examples_count": len(examples),
        }
        
        self.logger.log(f"PPO training stats: {stats_dict}")
        
        return stats_dict


# Choose which RL algorithm to use based on configuration
def get_rl_trainer():
    """
    Get the appropriate RL trainer based on configuration.
    
    Returns:
        RL trainer instance (TRR or PPO)
    """
    config = get_config()
    rl_algorithm = config.get("training.rl_algorithm", "TRR++")
    
    if rl_algorithm.upper() == "PPO":
        return PPOTrainingWrapper()
    else:
        # Default to TRR++
        model_service = get_model_service()
        return TRR(model_service.tokenizer) 