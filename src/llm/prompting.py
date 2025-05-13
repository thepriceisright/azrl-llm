from typing import Dict, List, Tuple, Any, Optional, Union
import json

from utils import get_logger, get_config


class PromptTemplates:
    """Templates for the different prompting tasks."""
    
    # Proposer Task Type Templates
    ABDUCTION_PROPOSER = """You are helping me generate a Python programming task. 
I want you to propose:
1. A Python function 'solution' that takes one or more inputs and returns an output
2. An example output value that would be returned by the function

The task should involve figuring out what input to the function would produce the given output.
Make sure the function has interesting logic that requires reasoning.

For reference, here are some examples of such tasks:
{reference_examples}

Please provide a function and an output. I'll try to figure out what input would produce that output.

Your response should be in this exact format:
```python
def solution(input_param):
    # Function implementation
    return result
```

Expected output: <the expected output value>

Important:
- The function must be deterministic (same input always gives same output)
- Use only standard Python libraries that don't require imports
- The function should be non-trivial but also not extremely complex
- The solution should be unambiguous (exactly one correct input should produce the output)
"""

    DEDUCTION_PROPOSER = """You are helping me generate a Python programming task. 
I want you to propose:
1. A Python function 'solution' that takes one or more inputs and returns an output
2. An example input value for that function

The task will involve figuring out what output the function would produce for the given input.
Make sure the function has interesting logic that requires reasoning.

For reference, here are some examples of such tasks:
{reference_examples}

Please provide a function and an input. I'll try to determine what output the function would produce.

Your response should be in this exact format:
```python
def solution(input_param):
    # Function implementation
    return result
```

Input: <the input value>

Important:
- The function must be deterministic (same input always gives same output)
- Use only standard Python libraries that don't require imports
- The function should be non-trivial but also not extremely complex
- The solution should be clear and demonstrate interesting logic
"""

    INDUCTION_PROPOSER = """You are helping me generate a Python programming task.
I want you to propose:
1. A set of example input-output pairs
2. A metaprompt (task description) that describes what function should be inferred from the examples

The task will involve figuring out what function would produce the given outputs for the corresponding inputs.
Make sure the examples demonstrate a pattern that requires reasoning to identify.

For reference, here are some examples of such tasks:
{reference_examples}

Please provide a set of example pairs and a metaprompt. I'll try to induce the function that generated them.

Your response should be in this exact format:
Example pairs:
Input: <input1>
Output: <output1>

Input: <input2>
Output: <output2>

... (provide at least 3 examples)

Metaprompt: <description of the task/function to be inferred>

Important:
- The pattern should be deterministic (same input always gives same output)
- Use only standard Python operations that don't require imports
- The pattern should be non-trivial but also not extremely complex
- The examples should be sufficient to uniquely identify the pattern
"""

    # Solver Task Type Templates
    ABDUCTION_SOLVER = """I have a Python function and its expected output. Your task is to figure out what input to the function would produce that output.

Here's the function:
```python
{program}
```

The output of this function should be:
{output}

Please determine what input to pass to the function to get this output. Show your reasoning, then provide the final answer in the format "Input: <your_answer>".
"""

    DEDUCTION_SOLVER = """I have a Python function and an input value. Your task is to figure out what output the function would produce for this input.

Here's the function:
```python
{program}
```

Input to the function:
{input}

Please determine what output the function would produce for this input. Show your reasoning, then provide the final answer in the format "Output: <your_answer>".
"""

    INDUCTION_SOLVER = """I have a set of example input-output pairs. Your task is to figure out the underlying function that produces these outputs from the inputs.

Here are the examples:
{examples}

Task description: {metaprompt}

Please determine the function that would generate these outputs for the given inputs. Provide your solution as a Python function named "solution".

Your response should include the full function definition in this format:
```python
def solution(input_param):
    # Your implementation here
    return result
```
"""


class PromptManager:
    """
    Manager for generating prompts for the different roles and task types.
    """
    def __init__(self):
        """Initialize the prompt manager."""
        self.logger = get_logger("prompt_manager")
        self.config = get_config()
        self.templates = PromptTemplates()
    
    def _format_reference_examples(self, examples: List[Dict[str, Any]], task_type: str) -> str:
        """
        Format reference examples for inclusion in a prompt.
        
        Args:
            examples: List of reference examples from the buffer
            task_type: Type of task (abduction, deduction, induction)
            
        Returns:
            Formatted reference examples as a string
        """
        if not examples:
            return "No reference examples available."
        
        formatted_examples = []
        
        if task_type == "abduction":
            for i, example in enumerate(examples):
                formatted = f"Example {i+1}:\n"
                formatted += f"```python\n{example['program']}\n```\n\n"
                formatted += f"Expected output: {example['output']}\n"
                formatted_examples.append(formatted)
        
        elif task_type == "deduction":
            for i, example in enumerate(examples):
                formatted = f"Example {i+1}:\n"
                formatted += f"```python\n{example['program']}\n```\n\n"
                formatted += f"Input: {example['input']}\n"
                formatted_examples.append(formatted)
        
        elif task_type == "induction":
            for i, example in enumerate(examples):
                formatted = f"Example {i+1}:\n"
                formatted += "Example pairs:\n"
                for input_val, output_val in example['examples']:
                    formatted += f"Input: {input_val}\n"
                    formatted += f"Output: {output_val}\n\n"
                formatted += f"Metaprompt: {example['metaprompt']}\n"
                formatted_examples.append(formatted)
        
        return "\n\n".join(formatted_examples)
    
    def format_examples_for_induction(self, examples: List[Tuple[Any, Any]]) -> str:
        """
        Format input-output examples for the induction solver.
        
        Args:
            examples: List of (input, output) tuples
            
        Returns:
            Formatted examples as a string
        """
        formatted = []
        for input_val, output_val in examples:
            formatted.append(f"Input: {input_val}\nOutput: {output_val}")
        
        return "\n\n".join(formatted)
    
    def generate_proposer_prompt(self, 
                               task_type: str,
                               reference_examples: List[Dict[str, Any]]) -> str:
        """
        Generate a prompt for the Proposer role.
        
        Args:
            task_type: Type of task (abduction, deduction, induction)
            reference_examples: List of reference examples from the buffer
            
        Returns:
            Formatted prompt
        """
        # Format reference examples
        formatted_references = self._format_reference_examples(reference_examples, task_type)
        
        # Get appropriate template
        if task_type == "abduction":
            template = self.templates.ABDUCTION_PROPOSER
        elif task_type == "deduction":
            template = self.templates.DEDUCTION_PROPOSER
        elif task_type == "induction":
            template = self.templates.INDUCTION_PROPOSER
        else:
            raise ValueError(f"Unknown task type: {task_type}")
        
        # Fill in template
        prompt = template.format(reference_examples=formatted_references)
        
        return prompt
    
    def generate_solver_prompt(self, task_type: str, task_data: Dict[str, Any]) -> str:
        """
        Generate a prompt for the Solver role.
        
        Args:
            task_type: Type of task (abduction, deduction, induction)
            task_data: Data for the task
            
        Returns:
            Formatted prompt
        """
        if task_type == "abduction":
            prompt = self.templates.ABDUCTION_SOLVER.format(
                program=task_data["program"],
                output=task_data["output"]
            )
        
        elif task_type == "deduction":
            prompt = self.templates.DEDUCTION_SOLVER.format(
                program=task_data["program"],
                input=task_data["input"]
            )
        
        elif task_type == "induction":
            # Format examples
            formatted_examples = self.format_examples_for_induction(task_data["examples"])
            
            prompt = self.templates.INDUCTION_SOLVER.format(
                examples=formatted_examples,
                metaprompt=task_data["metaprompt"]
            )
        
        else:
            raise ValueError(f"Unknown task type: {task_type}")
        
        return prompt


# Singleton instance
_prompt_manager = None


def get_prompt_manager() -> PromptManager:
    """
    Get or create the singleton prompt manager instance.
    
    Returns:
        PromptManager instance
    """
    global _prompt_manager
    if _prompt_manager is None:
        _prompt_manager = PromptManager()
    return _prompt_manager 