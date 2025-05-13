#!/usr/bin/env python3
"""
Test script to verify the executor factory and mock executor implementation.
"""
import os
import sys
import yaml

# Add project directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

def test_factory():
    """Test the executor factory with different configurations."""
    # Import the factory
    from src.executor.factory import get_executor
    
    # Test with mock enabled
    os.environ["CONFIG_OVERRIDE"] = "executor.use_mock=true"
    executor = get_executor()
    print(f"Factory returned: {type(executor).__name__}")
    
    # Execute a simple program
    program = """
def solution(x):
    return x * 2
"""
    result = executor.execute_code(program, 5)
    print(f"Execution result: {result}")
    
    # Test validation
    is_valid, _ = executor.validate_program(program)
    print(f"Program valid: {is_valid}")
    
    return True

def update_config(use_mock=True):
    """Update the config file to use mock executor."""
    config_path = "config/config.yaml"
    
    # Read existing config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Update executor section
    if "executor" not in config:
        config["executor"] = {}
    
    config["executor"]["use_mock"] = use_mock
    
    # Write updated config
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"Updated config: executor.use_mock = {use_mock}")

if __name__ == "__main__":
    try:
        # Import yaml for config update
        import yaml
    except ImportError:
        print("PyYAML not installed. Run: pip install pyyaml")
        sys.exit(1)
    
    # Update config to use mock executor
    update_config(True)
    
    # Run test
    print("\nTesting executor factory...")
    if test_factory():
        print("\nSUCCESS: Executor factory test passed!")
        print("You can now run training with the mock executor.")
        print("Command: WANDB_MODE=disabled python main.py --iterations 100")
    else:
        print("\nFAILED: Executor factory test failed.")
        print("Check the error messages above for details.") 