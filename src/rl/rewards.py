from typing import Dict, List, Tuple, Any, Optional, Union

from utils import get_logger, get_config


class RewardCalculator:
    """
    Calculates rewards for the Proposer and Solver roles in the AZR training loop.
    """
    def __init__(self):
        """Initialize the reward calculator."""
        self.logger = get_logger("reward_calculator")
        self.config = get_config()
        
        # Load reward parameters from config
        self.format_penalty_formatted = self.config.get("training.format_penalty.formatted_wrong", -0.5)
        self.format_penalty_error = self.config.get("training.format_penalty.format_error", -1.0)
    
    def calculate_proposer_reward(self, solver_results: List[Dict[str, Any]]) -> float:
        """
        Calculate the reward for the Proposer based on the solver's success rate.
        This implements the learnability reward from the paper.
        
        Args:
            solver_results: List of solver result dictionaries, each containing at least:
                - is_correct: Boolean indicating if solver was correct
                - has_format_error: Boolean indicating if there was a format error
        
        Returns:
            The calculated reward value
        """
        if not solver_results:
            return 0.0
        
        # Extract results
        num_rollouts = len(solver_results)
        num_correct = sum(1 for result in solver_results if result['is_correct'])
        
        # Calculate accuracy rate (learnability)
        accuracy_rate = num_correct / num_rollouts if num_rollouts > 0 else 0
        
        # The reward is the accuracy rate (0 to 1)
        # This follows Equation 4 in the paper
        reward = accuracy_rate
        
        self.logger.log(f"Proposer reward: {reward:.4f} (accuracy rate: {accuracy_rate:.4f}, "
                       f"{num_correct}/{num_rollouts} correct)")
        
        return reward
    
    def calculate_solver_reward(self,
                             is_correct: bool,
                             has_format_error: bool) -> float:
        """
        Calculate the reward for the Solver based on correctness and format.
        This implements the accuracy reward from the paper.
        
        Args:
            is_correct: Whether the solver's solution is correct
            has_format_error: Whether there was a format error in the solution
            
        Returns:
            The calculated reward value
        """
        # Base reward is binary: 1 for correct, 0 for incorrect
        # This follows Equation 5 in the paper
        base_reward = 1.0 if is_correct else 0.0
        
        # Apply format penalty if needed
        # This follows Equation 6 in the paper
        if has_format_error:
            # Wrong format error (could not parse solution)
            format_penalty = self.format_penalty_error
        elif not is_correct:
            # Wrong answer but correct format
            format_penalty = self.format_penalty_formatted
        else:
            # Correct answer with correct format
            format_penalty = 0.0
        
        # Final reward
        reward = base_reward + format_penalty
        
        self.logger.log(f"Solver reward: {reward:.4f} (correct: {is_correct}, "
                       f"format error: {has_format_error})")
        
        return reward


# Singleton instance
_reward_calculator = None


def get_reward_calculator() -> RewardCalculator:
    """
    Get or create the singleton reward calculator instance.
    
    Returns:
        RewardCalculator instance
    """
    global _reward_calculator
    if _reward_calculator is None:
        _reward_calculator = RewardCalculator()
    return _reward_calculator 