#!/usr/bin/env python3
"""
Training status monitor for AZRL.

This script scans the log files to provide a summary of the training status,
including current iteration, progress, performance metrics, and estimated time of completion.
"""
import os
import re
import sys
import glob
import time
import datetime
from collections import defaultdict

# Default log directory
DEFAULT_LOG_DIR = "logs"
# Pattern to find log files
LOG_FILE_PATTERN = "*.log"
# Number of lines to read from the end of each log file
TAIL_LINES = 500


def find_log_files(log_dir):
    """Find all log files in the given directory."""
    if not os.path.exists(log_dir):
        print(f"ERROR: Log directory {log_dir} does not exist.")
        return []
    
    log_files = glob.glob(os.path.join(log_dir, LOG_FILE_PATTERN))
    return sorted(log_files, key=os.path.getmtime, reverse=True)


def tail_file(file_path, n_lines=100):
    """Read the last n lines from a file."""
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            return lines[-n_lines:] if len(lines) > n_lines else lines
    except Exception as e:
        print(f"ERROR: Could not read file {file_path}: {e}")
        return []


def parse_timestamps(lines):
    """Extract timestamps from log lines."""
    timestamps = []
    timestamp_pattern = r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}'
    
    for line in lines:
        match = re.search(timestamp_pattern, line)
        if match:
            timestamp_str = match.group(0)
            try:
                timestamp = datetime.datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
                timestamps.append(timestamp)
            except ValueError:
                pass
    
    return timestamps


def extract_iteration_info(lines):
    """Extract iteration information from log lines."""
    current_iteration = None
    total_iterations = None
    iteration_pattern = r'Starting iteration (\d+)/(\d+)'
    completed_pattern = r'Completed (\d+)/(\d+) iterations'
    
    for line in lines:
        match = re.search(iteration_pattern, line)
        if match:
            current_iteration = int(match.group(1))
            total_iterations = int(match.group(2))
        
        match = re.search(completed_pattern, line)
        if match:
            current_iteration = int(match.group(1))
            total_iterations = int(match.group(2))
    
    return current_iteration, total_iterations


def extract_metrics(lines):
    """Extract performance metrics from log lines."""
    metrics = {
        'proposer_reward': [],
        'solver_accuracy': [],
        'duration': []
    }
    
    # Patterns to match metrics
    proposer_pattern = r'proposer_reward=(\d+\.\d+)'
    solver_pattern = r'solver_accuracy=(\d+\.\d+)'
    duration_pattern = r'completed in (\d+\.\d+) seconds'
    
    for line in lines:
        # Extract proposer reward
        match = re.search(proposer_pattern, line)
        if match:
            metrics['proposer_reward'].append(float(match.group(1)))
        
        # Extract solver accuracy
        match = re.search(solver_pattern, line)
        if match:
            metrics['solver_accuracy'].append(float(match.group(1)))
        
        # Extract iteration duration
        match = re.search(duration_pattern, line)
        if match:
            metrics['duration'].append(float(match.group(1)))
    
    return metrics


def extract_eta(lines):
    """Extract estimated time remaining from log lines."""
    eta_pattern = r'Estimated time remaining: (\d+)h (\d+)m (\d+)s'
    
    for line in reversed(lines):  # Start from the most recent lines
        match = re.search(eta_pattern, line)
        if match:
            hours = int(match.group(1))
            minutes = int(match.group(2))
            seconds = int(match.group(3))
            
            # Calculate total seconds
            total_seconds = hours * 3600 + minutes * 60 + seconds
            
            # Create a timedelta object
            eta = datetime.timedelta(seconds=total_seconds)
            eta_datetime = datetime.datetime.now() + eta
            
            return eta, eta_datetime
    
    return None, None


def extract_status_messages(lines):
    """Extract status messages from log lines."""
    status_messages = []
    status_pattern = r'Current status: (.+)$'
    heartbeat_pattern = r'❤️ HEARTBEAT ❤️'
    
    for line in lines:
        if re.search(heartbeat_pattern, line):
            match = re.search(status_pattern, line)
            if match:
                status_messages.append(match.group(1))
    
    return status_messages


def format_duration(seconds):
    """Format duration in seconds as a human-readable string."""
    if seconds is None:
        return "N/A"
    
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"


def check_training_status(log_dir=DEFAULT_LOG_DIR):
    """Check the training status by analyzing log files."""
    # Find log files
    log_files = find_log_files(log_dir)
    if not log_files:
        print(f"No log files found in {log_dir}.")
        return
    
    print(f"Found {len(log_files)} log files in {log_dir}.")
    
    # Combine latest lines from all log files
    all_lines = []
    for log_file in log_files[:3]:  # Only read the 3 most recent files
        print(f"Reading {os.path.basename(log_file)}...")
        lines = tail_file(log_file, TAIL_LINES)
        all_lines.extend(lines)
    
    # Sort lines by timestamp if possible (this is a simplistic approach)
    all_lines.sort()
    
    # Extract information
    timestamps = parse_timestamps(all_lines)
    current_iteration, total_iterations = extract_iteration_info(all_lines)
    metrics = extract_metrics(all_lines)
    eta, eta_datetime = extract_eta(all_lines)
    status_messages = extract_status_messages(all_lines)
    
    # Output the status report
    print("\n=== AZRL Training Status Report ===")
    print(f"Time of report: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if timestamps:
        first_timestamp = min(timestamps)
        last_timestamp = max(timestamps)
        duration = last_timestamp - first_timestamp
        print(f"Training duration: {duration}")
        print(f"Latest activity: {last_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Check if training seems to be stalled (no activity in the last hour)
        time_since_last = datetime.datetime.now() - last_timestamp
        if time_since_last > datetime.timedelta(hours=1):
            print(f"WARNING: No activity detected in {time_since_last}. Training might be stalled.")
    
    # Print progress information
    if current_iteration is not None and total_iterations is not None:
        progress = (current_iteration / total_iterations) * 100
        print(f"Progress: {current_iteration}/{total_iterations} iterations ({progress:.1f}%)")
    
    # Print metrics
    print("\nPerformance Metrics:")
    for metric_name, values in metrics.items():
        if values:
            avg_value = sum(values) / len(values)
            recent_values = values[-min(5, len(values)):]  # Last 5 values
            recent_avg = sum(recent_values) / len(recent_values)
            
            print(f"  {metric_name}: {avg_value:.4f} (avg), {recent_avg:.4f} (recent)")
            
            if metric_name == 'duration':
                total_remaining = (total_iterations - current_iteration) * recent_avg if current_iteration is not None else None
                print(f"  Estimated remaining time: {format_duration(total_remaining)}")
    
    # Print ETA information
    if eta:
        print(f"\nEstimated completion time: {eta_datetime.strftime('%Y-%m-%d %H:%M:%S')} (in {eta})")
    
    # Print the latest status message
    if status_messages:
        print(f"\nLatest status: {status_messages[-1]}")
    
    print("\n=== End of Status Report ===")


if __name__ == "__main__":
    # Parse command-line arguments
    if len(sys.argv) > 1:
        log_dir = sys.argv[1]
    else:
        log_dir = DEFAULT_LOG_DIR
    
    check_training_status(log_dir) 