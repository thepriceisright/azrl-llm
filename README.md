# Absolute Zero Reinforcement Learning (AZRL)

An implementation of the Absolute Zero Reinforcement Learning approach based on the paper ["Absolute Zero: Reinforced Self-play Reasoning with Zero Data"](https://arxiv.org/html/2505.03335v2).

## Overview

This project implements the Absolute Zero Reasoner (AZR) framework, which enables a Large Language Model (LLM) to improve its reasoning capabilities through self-play without relying on external datasets. The framework involves the following components:

- **Proposer Role**: The LLM generates task proposals of varying types.
- **Solver Role**: The LLM attempts to solve the proposed tasks.
- **Secure Code Executor**: Validates, executes, and verifies Python code within a secure sandbox.
- **Task Buffer**: Stores and manages valid task examples.
- **RL Updater**: Updates the model weights based on task learnability and solution accuracy.

## Features

- Implementation of three task types: Abduction, Deduction, and Induction
- Secure Docker-based code execution environment
- TRR++ (Task-Relative REINFORCE++) implementation for RL updates
- Alternative PPO implementation using TRL library
- RunPod integration for running on cloud GPU

## Setup

### Requirements

- Python 3.10+
- PyTorch 2.0+
- Docker
- NVIDIA GPU with at least 24GB VRAM (ideally 48GB for A40)

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/thepriceisright/azrl-llm.git
   cd azrl-llm
   ```

2. Install the dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Setup the environment:
   ```
   bash setup_runpod.sh
   ```

## Running the Training

To start the training process:

```
python main.py --iterations 100
```

Additional options:
- `--config PATH`: Path to the configuration file (default: `config/config.yaml`)
- `--checkpoint PATH`: Path to a checkpoint to resume training
- `--seed INT`: Random seed for reproducibility (default: 42)

## Project Structure

- `src/buffer/`: Task buffer implementation
- `src/executor/`: Secure code execution environment
- `src/llm/`: LLM service and prompting utilities
- `src/orchestrator/`: Main training loop orchestration
- `src/rl/`: Reinforcement learning algorithms (TRR++, PPO)
- `utils/`: Utility functions for logging, configuration, etc.
- `config/`: Configuration files
- `logs/`: Training logs
- `main.py`: Main entry point

## Configuration

The main configuration file is located at `config/config.yaml`. It includes settings for:

- Model configuration
- Training hyperparameters
- Executor settings
- Task types
- Logging options
- Path configurations

## RunPod Deployment

This implementation is designed to run on RunPod with A40 GPUs. The `setup_runpod.sh` script sets up the environment for RunPod deployment. Make sure you have a Network Volume attached to your RunPod instance for persistent storage.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- ["Absolute Zero: Reinforced Self-play Reasoning with Zero Data"](https://arxiv.org/html/2505.03335v2) paper by Zhao et al.
- [RunPod](https://docs.runpod.io/overview) for cloud GPU infrastructure
- [Qwen](https://huggingface.co/Qwen) for the base LLM model 