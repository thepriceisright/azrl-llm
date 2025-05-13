# RunPod Deployment Guide

This guide explains how to deploy the AZRL implementation to RunPod for training.

## Prerequisites

- A RunPod account with credits available
- A GPU template with PyTorch and at least 24GB VRAM (A40 recommended)

## Setting up RunPod

1. **Create a Network Volume**:
   - Go to Storage section in RunPod dashboard
   - Click "Create Volume"
   - Select a region close to you
   - Set a name and size (20GB or more recommended)

2. **Launch a GPU Pod**:
   - Select an A40 GPU (as recommended)
   - Choose a template with PyTorch and CUDA
   - Attach your network volume
   - Set volume mount path to `/workspace`

3. **Connect to your pod**:
   - Use the web terminal or SSH
   - Navigate to your volume: `cd /workspace`

## Deploying the Code

1. **Clone the repository**:
   ```bash
   cd /workspace
   git clone https://github.com/yourusername/azrl-llm.git
   cd azrl-llm
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Initialize the workspace**:
   ```bash
   ./init_workspace.sh
   ```

## Running Training

1. **Verify configuration**:
   - Open `config/config.yaml`
   - Ensure `executor.use_mock` is set to `true`
   - Adjust any other settings as needed

2. **Start training**:
   ```bash
   cd /workspace/azrl-llm
   WANDB_MODE=disabled python main.py --iterations 100
   ```

   If you want to use Weights & Biases:
   ```bash
   WANDB_API_KEY=your_api_key python main.py --iterations 100
   ```

## Running Training in the Background

To keep training running even after disconnecting from SSH or closing the browser:

1. **Using tmux** (recommended):
   ```bash
   apt-get update && apt-get install -y tmux
   tmux new -s azrl_training
   
   # Inside tmux session:
   cd /workspace/azrl-llm
   WANDB_MODE=disabled python main.py --iterations 100
   
   # Detach from session with Ctrl+b then d
   # Later reconnect with:
   tmux attach -t azrl_training
   ```

2. **Using nohup**:
   ```bash
   cd /workspace/azrl-llm
   nohup python main.py --iterations 100 > training.log 2>&1 &
   
   # Check progress with:
   tail -f training.log
   ```

## Troubleshooting

- **Docker-related errors**: Ensure `executor.use_mock` is set to `true` in config
- **CUDA out of memory**: Reduce batch size in config
- **Missing directories**: Run `init_workspace.sh` again

## Monitoring and Results

- Training checkpoints will be saved to `/workspace/checkpoints`
- Logs are stored in `/workspace/logs`
- Task buffer data is in `/workspace/buffer`

These directories persist on your network volume even after stopping your pod. 