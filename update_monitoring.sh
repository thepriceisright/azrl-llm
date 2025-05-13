#!/bin/bash
# Script to update an existing AZRL installation with improved monitoring features

set -e  # Exit on any error

# Backup existing files
echo "Creating backups of existing files..."
cp main.py main.py.bak
cp src/orchestrator/azr_pipeline.py src/orchestrator/azr_pipeline.py.bak

# Add progress_interval to config if it doesn't exist
if ! grep -q "progress_interval" config/config.yaml; then
    echo "Adding logging.progress_interval to config..."
    echo "" >> config/config.yaml
    echo "# Added by update_monitoring.sh" >> config/config.yaml
    echo "logging:" >> config/config.yaml
    echo "  progress_interval: 60  # Seconds between progress updates" >> config/config.yaml
fi

# Apply monitoring changes to config
echo "Updating config with monitoring settings..."
if grep -q "logging:" config/config.yaml; then
    sed -i '/logging:/a\  progress_interval: 60  # Seconds between progress updates' config/config.yaml
fi

# Install monitoring scripts
echo "Installing monitoring scripts..."
curl -s https://raw.githubusercontent.com/YOUR_USERNAME/azrl-llm/monitoring/check_training_status.py > check_training_status.py
chmod +x check_training_status.py

# Modify main.py and azr_pipeline.py with the monitoring improvements
# This is simplified - in a real update script, you might use patch files or more robust approaches
echo "Updating main.py with heartbeat thread..."
curl -s https://raw.githubusercontent.com/YOUR_USERNAME/azrl-llm/monitoring/main.py > main.py

echo "Updating azr_pipeline.py with detailed progress logging..."
curl -s https://raw.githubusercontent.com/YOUR_USERNAME/azrl-llm/monitoring/src/orchestrator/azr_pipeline.py > src/orchestrator/azr_pipeline.py

echo ""
echo "===== Monitoring Update Complete ====="
echo ""
echo "New features:"
echo "1. Detailed step-by-step progress logging"
echo "2. Regular heartbeat messages every 5 minutes (configurable with --heartbeat)"
echo "3. Activity timestamps and duration tracking"
echo "4. Estimated time remaining calculations"
echo "5. Status monitoring script (./check_training_status.py)"
echo ""
echo "To check the status of a running training job:"
echo "  ./check_training_status.py /path/to/logs"
echo ""
echo "To start training with monitoring:"
echo "  WANDB_MODE=disabled python main.py --iterations 100 --heartbeat 300"
echo ""
echo "Restore from backup if needed:"
echo "  cp main.py.bak main.py"
echo "  cp src/orchestrator/azr_pipeline.py.bak src/orchestrator/azr_pipeline.py" 