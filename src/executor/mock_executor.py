import re
import ast
import json
from typing import Dict, Any, List, Optional, Tuple, Union

from utils import get_logger, get_config


class MockCodeExecutor:
    """
    Mock implementation of the code executor for testing purposes.
    This version uses Python's exec() directly instead of Docker.
    CAUTION: This is NOT secure and should only be used for testing, not in production.
    """
    def __init__(self):
        """Initialize the mock code executor."""
        self.logger = get_logger("mock_executor")
        self.config = get_config()
        
        # Get configuration
        self.forbidden_modules = self.config.get("executor.forbidden_modules", [])
    
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
        Execute Python code directly (no Docker, NOT secure!).
        
        Args:
            code: The Python code to execute.
            input_data: Input data for the program.
            
        Returns:
            Dictionary containing the result of execution.
        """
        # Directly execute the code in the current Python process
        result = {
            "success": False,
            "output": None,
            "error": None
        }
        
        try:
            # Check for safety first (even though this is a mock)
            is_safe, reason = self._check_code_safety(code)
            if not is_safe:
                result["error"] = reason
                return result
            
            # Create a local namespace for execution
            local_vars = {"__input": input_data}
            
            # Execute the code
            exec(code, {}, local_vars)
            
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
                
                result["output"] = output
                result["success"] = True
            else:
                # Just return whatever's in the local namespace as output
                result["output"] = local_vars.get("output", None)
                result["success"] = True
        except Exception as e:
            result["error"] = str(e)
            result["success"] = False
        
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
_mock_executor = None


def get_mock_executor() -> MockCodeExecutor:
    """
    Get or create the singleton mock executor instance.
    
    Returns:
        MockCodeExecutor instance
    """
    global _mock_executor
    if _mock_executor is None:
        _mock_executor = MockCodeExecutor()
    return _mock_executor 