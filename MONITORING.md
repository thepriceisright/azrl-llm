# AZRL Monitoring Guide

This document explains the monitoring enhancements for the AZRL implementation.

## Overview

The monitoring enhancements address the issue of training jobs appearing to hang or stall by providing:

1. Detailed step-by-step progress logging
2. Regular heartbeat messages
3. Activity timestamps and estimated completion times
4. A status monitoring script to check on running jobs

## Features

### 1. Enhanced Progress Logging

The training process now logs detailed information about each step:

- Timestamped log entries with [YYYY-MM-DD HH:MM:SS] prefix
- Step-by-step progress indicators ([STEP 1/6], [STEP 2/6], etc.)
- Duration tracking for each step and iteration
- Periodic progress updates even during long-running operations

Example:
```
[2025-05-13 01:32:47] Starting iteration 3/100
[STEP 1/6] Generating task proposal for deduction
Proposal generation completed in 12.45 seconds
[STEP 2/6] Validating task proposal
...
```

### 2. Heartbeat Thread

A dedicated heartbeat thread outputs status messages at regular intervals:

- Shows that the process is still running
- Displays the current operation being performed
- Includes elapsed time since training started
- Default interval is 5 minutes (300 seconds), configurable with `--heartbeat` flag

Example:
```
[2025-05-13 01:45:22] ❤️ HEARTBEAT ❤️ Process running for 1h 23m 15s
Current status: Running solver 3/8 to estimate learnability
```

### 3. Training Status Monitor

A dedicated script (`check_training_status.py`) allows you to check the status of a running job at any time:

- Scan log files to extract current progress
- Show performance metrics (proposer reward, solver accuracy, etc.)
- Calculate estimated time of completion
- Check for potential stalls or issues

## Usage

### Starting Training with Monitoring

```bash
# With default 5-minute heartbeat interval
WANDB_MODE=disabled python main.py --iterations 100

# With custom heartbeat interval (1 minute)
WANDB_MODE=disabled python main.py --iterations 100 --heartbeat 60
```

### Checking Training Status

```bash
# Check status using default log directory
./check_training_status.py

# Check status in a custom log directory
./check_training_status.py /workspace/logs
```

### In a tmux Session

```bash
# Start a tmux session
tmux new -s azrl_training

# Start training with monitoring
cd /workspace/azrl-llm
WANDB_MODE=disabled python main.py --iterations 100

# Detach from session with Ctrl+b then d
# Later reconnect and check status:
tmux attach -t azrl_training

# Or check status without reconnecting:
./check_training_status.py /workspace/logs
```

## Configuration

The monitoring features can be configured in `config/config.yaml`:

```yaml
logging:
  level: "INFO"
  use_wandb: true
  wandb_project: "azrl-qwen25-coder-3b"
  log_dir: "logs"
  progress_interval: 60  # Seconds between progress updates
```

## Troubleshooting

If the training appears to be stalled:

1. Run `./check_training_status.py` to see the latest activity
2. Check if heartbeat messages are still appearing in the logs
3. Look for error messages or exceptions in the log files
4. Check system resources (GPU memory, CPU, disk space)

If the training has genuinely stalled, you may need to restart it, potentially with a smaller batch size or other adjustments.

## Implementation Details

The monitoring improvements are implemented in:

- `main.py`: Heartbeat thread and status tracking
- `src/orchestrator/azr_pipeline.py`: Enhanced progress logging
- `check_training_status.py`: Status monitoring utility

These changes maintain full compatibility with the existing codebase while providing much better visibility into the training process. 