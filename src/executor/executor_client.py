import os
import json
import docker
import tempfile
import ast
import re
import time
from typing import Dict, Any, List, Optional, Tuple, Union
import hashlib

from utils import get_logger, get_config


class CodeExecutor:
    """
    Client for the secure code executor. Manages Docker containers and provides
    an interface for executing code securely.
    """
    def __init__(self):
        """Initialize the code executor."""
        self.logger = get_logger("executor")
        self.config = get_config()
        
        # Get configuration
        self.timeout = self.config.get("executor.timeout", 15)
        self.memory_limit = self.config.get("executor.memory_limit", "1g")
        self.cpu_limit = self.config.get("executor.cpu_limit", "1.0")
        self.forbidden_modules = self.config.get("executor.forbidden_modules", [])
        
        # Initialize Docker client
        self.docker_client = docker.from_env()
        
        # Build or pull the executor image
        self._setup_executor_image()
    
    def _setup_executor_image(self) -> None:
        """Setup the executor Docker image."""
        self.image_name = "azrl-executor:latest"
        
        try:
            # Check if image already exists
            self.docker_client.images.get(self.image_name)
            self.logger.log(f"Executor image {self.image_name} already exists.")
        except docker.errors.ImageNotFound:
            # Build the image
            self.logger.log(f"Building executor image {self.image_name}...")
            
            # Get the path to the Dockerfile
            script_dir = os.path.dirname(os.path.abspath(__file__))
            dockerfile_path = os.path.join(script_dir)
            
            # Build the image
            try:
                self.docker_client.images.build(
                    path=dockerfile_path,
                    tag=self.image_name,
                    rm=True
                )
                self.logger.log(f"Successfully built executor image {self.image_name}.")
            except docker.errors.BuildError as e:
                self.logger.log(f"Failed to build executor image: {e}", level="ERROR")
                raise
    
    def _check_code_safety(self, code: str) -> Tuple[bool, str]:
        """
        Check if the code is safe to execute by looking for forbidden modules.
        
        Args:
            code: The Python code to check.
            
        Returns:
            Tuple of (is_safe, reason)
        """
        try:
            tree = ast.parse(code)
            
            # Check for import statements
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for name in node.names:
                        if name.name in self.forbidden_modules:
                            return False, f"Forbidden module: {name.name}"
                
                elif isinstance(node, ast.ImportFrom):
                    if node.module in self.forbidden_modules:
                        return False, f"Forbidden module: {node.module}"
                    
                    # Check for imports from submodules
                    for forbidden in self.forbidden_modules:
                        if node.module and node.module.startswith(f"{forbidden}."):
                            return False, f"Forbidden module: {node.module}"
            
            # Additional regex checks for dynamic imports
            for forbidden in self.forbidden_modules:
                # Check for __import__(module)
                if re.search(rf"__import__\s*\(\s*['\"]({forbidden}|{forbidden}\.[^'\"]*)['\"]", code):
                    return False, f"Dynamic import of forbidden module: {forbidden}"
                
                # Check for importlib.import_module(module)
                if re.search(rf"import_module\s*\(\s*['\"]({forbidden}|{forbidden}\.[^'\"]*)['\"]", code):
                    return False, f"Dynamic import of forbidden module: {forbidden}"
            
            return True, ""
        except SyntaxError as e:
            return False, f"Syntax error: {str(e)}"
    
    def validate_program(self, code: str) -> Tuple[bool, str]:
        """
        Validate a program for syntax and safety.
        
        Args:
            code: The Python code to validate.
            
        Returns:
            Tuple of (is_valid, reason)
        """
        # Check for syntax errors
        try:
            ast.parse(code)
        except SyntaxError as e:
            return False, f"Syntax error: {str(e)}"
        
        # Check for safety
        is_safe, reason = self._check_code_safety(code)
        if not is_safe:
            return False, reason
        
        return True, ""
    
    def check_determinism(self, code: str, input_data: Any) -> Tuple[bool, Any, Any]:
        """
        Check if a program is deterministic by running it twice and comparing the outputs.
        
        Args:
            code: The Python code to check.
            input_data: Input data for the program.
            
        Returns:
            Tuple of (is_deterministic, output1, output2)
        """
        # Run the program twice
        result1 = self.execute_code(code, input_data)
        result2 = self.execute_code(code, input_data)
        
        # Check if both runs succeeded
        if not result1["success"] or not result2["success"]:
            return False, result1, result2
        
        # Compare the outputs
        try:
            # Convert to string for comparison if needed
            output1_str = json.dumps(result1["output"], sort_keys=True)
            output2_str = json.dumps(result2["output"], sort_keys=True)
            
            is_deterministic = output1_str == output2_str
            return is_deterministic, result1["output"], result2["output"]
        except (TypeError, ValueError):
            # If outputs can't be converted to JSON, compare directly
            is_deterministic = result1["output"] == result2["output"]
            return is_deterministic, result1["output"], result2["output"]
    
    def execute_code(self, code: str, input_data: Any = None) -> Dict[str, Any]:
        """
        Execute Python code securely in a Docker container.
        
        Args:
            code: The Python code to execute.
            input_data: Input data for the program.
            
        Returns:
            Dictionary containing the result of execution.
        """
        # Create a temporary file for input
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False) as temp_file:
            # Write the input JSON to the file
            json.dump({
                "code": code,
                "input": input_data
            }, temp_file)
            temp_file_path = temp_file.name
        
        try:
            # Generate a unique container name
            container_name = f"azrl-executor-{hashlib.md5(code.encode()).hexdigest()[:10]}-{int(time.time())}"
            
            # Run the container
            container = self.docker_client.containers.run(
                image=self.image_name,
                name=container_name,
                volumes={temp_file_path: {'bind': '/sandbox/input.json', 'mode': 'ro'}},
                command=["python", "/sandbox/executor.py", "<", "/sandbox/input.json"],
                detach=True,
                remove=True,
                network_mode="none",
                mem_limit=self.memory_limit,
                cpu_quota=int(float(self.cpu_limit) * 100000),
                cpu_period=100000
            )
            
            # Wait for the container to finish or timeout
            try:
                container.wait(timeout=self.timeout)
                logs = container.logs().decode('utf-8')
                
                # Parse the output
                try:
                    result = json.loads(logs)
                except json.JSONDecodeError:
                    result = {
                        "success": False,
                        "error": "Failed to parse output from executor",
                        "stdout": logs
                    }
            except Exception as e:
                # Handle timeout or other errors
                try:
                    # Try to stop and remove the container
                    container.stop(timeout=1)
                    container.remove(force=True)
                except Exception:
                    pass
                
                result = {
                    "success": False,
                    "error": f"Execution error: {str(e)}",
                    "timeout": True
                }
        finally:
            # Clean up the temporary file
            try:
                os.unlink(temp_file_path)
            except Exception:
                pass
        
        return result
    
    def verify_solution(self, 
                       task_type: str, 
                       program: str, 
                       solution: Union[str, Any], 
                       expected_output: Any, 
                       input_data: Any = None) -> Tuple[bool, Dict[str, Any]]:
        """
        Verify a solution for a given task.
        
        Args:
            task_type: Type of task (abduction, deduction, induction).
            program: The program code.
            solution: The proposed solution.
            expected_output: The expected output.
            input_data: Input data for abduction tasks.
            
        Returns:
            Tuple of (is_correct, details)
        """
        if task_type == "deduction":
            # For deduction, compare the solution directly with the expected output
            try:
                # Convert to string for comparison if needed
                solution_str = json.dumps(solution, sort_keys=True)
                expected_str = json.dumps(expected_output, sort_keys=True)
                
                is_correct = solution_str == expected_str
            except (TypeError, ValueError):
                # If outputs can't be converted to JSON, compare directly
                is_correct = solution == expected_output
            
            return is_correct, {"solution": solution, "expected": expected_output}
        
        elif task_type == "abduction":
            # For abduction, execute the program with the solution input and check if it matches the expected output
            result = self.execute_code(program, solution)
            
            if not result["success"]:
                return False, {
                    "error": result.get("error", "Execution failed"),
                    "solution_input": solution,
                    "expected_output": expected_output
                }
            
            try:
                # Convert to string for comparison if needed
                output_str = json.dumps(result["output"], sort_keys=True)
                expected_str = json.dumps(expected_output, sort_keys=True)
                
                is_correct = output_str == expected_str
            except (TypeError, ValueError):
                # If outputs can't be converted to JSON, compare directly
                is_correct = result["output"] == expected_output
            
            return is_correct, {
                "solution_input": solution,
                "actual_output": result["output"],
                "expected_output": expected_output
            }
        
        elif task_type == "induction":
            # For induction, solution is a program that should match the expected program behavior
            # We need to test it on input_data which should be a list of (input, output) pairs
            if not input_data or not isinstance(input_data, list):
                return False, {"error": "Invalid input data for induction task"}
            
            all_correct = True
            failures = []
            
            for i, (test_input, test_output) in enumerate(input_data):
                result = self.execute_code(solution, test_input)
                
                if not result["success"]:
                    all_correct = False
                    failures.append({
                        "index": i,
                        "input": test_input,
                        "expected": test_output,
                        "error": result.get("error", "Execution failed")
                    })
                    continue
                
                try:
                    # Convert to string for comparison if needed
                    output_str = json.dumps(result["output"], sort_keys=True)
                    expected_str = json.dumps(test_output, sort_keys=True)
                    
                    if output_str != expected_str:
                        all_correct = False
                        failures.append({
                            "index": i,
                            "input": test_input,
                            "expected": test_output,
                            "actual": result["output"]
                        })
                except (TypeError, ValueError):
                    # If outputs can't be converted to JSON, compare directly
                    if result["output"] != test_output:
                        all_correct = False
                        failures.append({
                            "index": i,
                            "input": test_input,
                            "expected": test_output,
                            "actual": result["output"]
                        })
            
            return all_correct, {
                "failures": failures,
                "num_tests": len(input_data),
                "num_failures": len(failures)
            }
        
        else:
            raise ValueError(f"Unknown task type: {task_type}")


# Singleton instance
_executor = None


def get_executor() -> CodeExecutor:
    """
    Get or create the singleton executor instance.
    
    Returns:
        CodeExecutor instance
    """
    global _executor
    if _executor is None:
        _executor = CodeExecutor()
    return _executor 