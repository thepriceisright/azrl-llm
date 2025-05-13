#!/usr/bin/env python
"""
Secure Python code executor for Absolute Zero Reinforcement Learning.

This script is intended to be run inside a Docker container to safely execute
user-generated code. It takes the code as input, executes it in a controlled
environment, and returns the result.
"""
import sys
import json
import traceback
import ast
from io import StringIO
from typing import Dict, Any, List, Optional, Tuple

# Redirect stdout and stderr to capture output
_stdout = sys.stdout
_stderr = sys.stderr


def execute_code(code: str, input_data: Any = None) -> Dict[str, Any]:
    """
    Execute the provided Python code with the given input data.
    
    Args:
        code: Python code to execute
        input_data: Input data to pass to the function
        
    Returns:
        Dictionary containing the result of execution
    """
    # Capture output
    stdout_capture = StringIO()
    stderr_capture = StringIO()
    sys.stdout = stdout_capture
    sys.stderr = stderr_capture
    
    result = {
        "success": False,
        "output": None,
        "error": None,
        "stdout": None,
        "stderr": None
    }
    
    try:
        # Compile the code
        compiled_code = compile(code, "<string>", "exec")
        
        # Create a local namespace for execution
        local_vars = {"__input": input_data}
        
        # Execute the code
        exec(compiled_code, {}, local_vars)
        
        # Check if the code defines a function
        if "solution" in local_vars and callable(local_vars["solution"]):
            # Execute the function with the input
            if input_data is not None:
                if isinstance(input_data, dict):
                    output = local_vars["solution"](**input_data)
                elif isinstance(input_data, list) or isinstance(input_data, tuple):
                    output = local_vars["solution"](*input_data)
                else:
                    output = local_vars["solution"](input_data)
            else:
                output = local_vars["solution"]()
        else:
            # Just return whatever's in the local namespace as output
            output = local_vars.get("output", None)
        
        result["success"] = True
        result["output"] = output
    except Exception as e:
        result["error"] = str(e)
        result["traceback"] = traceback.format_exc()
    finally:
        # Capture stdout and stderr
        result["stdout"] = stdout_capture.getvalue()
        result["stderr"] = stderr_capture.getvalue()
        
        # Restore stdout and stderr
        sys.stdout = _stdout
        sys.stderr = _stderr
    
    return result


def main():
    """Main entry point for the executor."""
    try:
        # Read input JSON from stdin
        input_json = json.loads(sys.stdin.read())
        
        code = input_json.get("code", "")
        input_data = input_json.get("input", None)
        
        # Execute the code
        result = execute_code(code, input_data)
        
        # Print the result as JSON to stdout
        print(json.dumps(result))
    except Exception as e:
        error_result = {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }
        print(json.dumps(error_result))


if __name__ == "__main__":
    main() 