**Technical Project Plan: Absolute Zero Training for Qwen2.5-Coder-3B**

**1. Introduction & Overview**

* **Project Goal:** To implement and experiment with the Absolute Zero (AZ) reinforcement learning paradigm, enabling a Large Language Model (LLM) to improve its reasoning capabilities through self-play without relying on external datasets.
* **Methodology:** Based on the "Absolute Zero Reasoner" (AZR) approach described in the paper "Absolute Zero: Reinforced Self-play Reasoning with Zero Data" (arXiv:2505.03335v2).
* **Target Model:** Qwen/Qwen2.5-Coder-3B (approx. 3 billion parameters).
* **Target Infrastructure:** RunPod Cloud GPU Platform, specifically utilizing NVIDIA A40 GPUs.

**2. Goals & Objectives**

* **Primary Goal:** Successfully implement and run a stable AZ training loop for the Qwen2.5-Coder-3B model on RunPod.
* **Key Objectives:**
    * Implement all core components of the AZR architecture (Proposer, Solver, Verifiable Environment, Task Buffer, RL Updater).
    * Implement the three specified code reasoning task types: Abduction, Deduction, Induction.
    * Demonstrate successful task generation, validation, solving, verification, and model updates within the loop.
    * Establish robust monitoring and logging to track training progress and system behavior.
    * (Stretch Goal) Observe measurable improvement in relevant coding or reasoning benchmarks compared to the base Qwen2.5-Coder-3B model after a significant number of training iterations.

**3. Scope**

* **In Scope:**
    * Setup and configuration of the Qwen2.5-Coder-3B model on RunPod.
    * Implementation of the core AZR self-play loop (Algorithm 1 in the paper).
    * Implementation of Abduction, Deduction, and Induction task types focused on Python code generation/reasoning.
    * Development of a secure Python code execution environment.
    * Implementation of the specified reward functions (Learnability, Accuracy, Format Penalty).
    * Implementation of the TRR++ RL update algorithm (or PPO as a fallback).
    * Initial deployment and testing on a single RunPod A40 node.
    * Basic monitoring and logging of the training process.
* **Out of Scope:**
    * Large-scale distributed training beyond a single node (initially).
    * Exploration of alternative verifiable environments (e.g., web interaction, simulators).
    * Advanced safety mechanisms beyond input filtering and sandbox restrictions.
    * Hyperparameter optimization beyond initial settings derived from the paper/model constraints.
    * Production deployment or serving of the trained model.

**4. System Architecture**

The system will consist of the following high-level components running within or coordinated by a primary RunPod instance:

1.  **Main Orchestrator:** A Python script managing the overall AZ loop (Algorithm 1), coordinating calls between other components.
2.  **LLM Service:** Loads and manages the Qwen2.5-Coder-3B model (using libraries like Hugging Face Transformers). Handles prompt formatting and inference calls for both Proposer and Solver roles.
3.  **Secure Code Executor:** An isolated environment (likely Docker-based within the pod) responsible for:
    * Receiving Python code snippets and inputs.
    * Validating code (syntax checks, safety checks against forbidden modules, determinism checks).
    * Executing validated code within strict resource limits (CPU, memory, time).
    * Returning execution results (output, errors, success/failure status).
    * This component will act as the "environment" providing verifiable rewards.
4.  **Task Buffer Storage:** Uses RunPod's Network Volume for persistent storage of validated task triplets `(p, i, o)` and induction task data `(p, {(i^n, o^n)}, m)`. Handles saving new tasks and sampling reference tasks for the proposer.
5.  **Monitoring & Logging:** Collects logs and metrics from all components, potentially using standard Python logging, file output to the network volume, or integration with tools like Weights & Biases.

**Diagrammatic Flow:**

```mermaid
graph TD
    A[Orchestrator] --> B(LLM Service - Proposer Role);
    B -- Generate Task Proposal (p, i) / (p, {i^n}, m) --> A;
    A -- Send Proposed Task to Executor --> C{Secure Code Executor};
    C -- Validate Task (Syntax, Safety, Determinism) --> A;
    subgraph Task Validation & Ground Truth Generation
        C -- If Valid, Execute p(i) -> o --> D[Task Buffer Storage];
    end
    A -- Add Validated Task to Buffer --> D;
    A -- Sample Valid Task (x) from Buffer --> E(LLM Service - Solver Role);
    E -- Generate Solution (y) --> A;
    A -- Send Solution (y) and Ground Truth (y*) to Executor --> F{Secure Code Executor};
    subgraph Answer Verification
        F -- Verify y against y* --> G[Reward Calculation];
    end
    G -- Calculate r_solve, r_propose, Format Penalties --> H(RL Updater - TRR++);
    H -- Update LLM Weights --> E;
    H --> B; %% Weights are shared
    D -- Sample K References --> B; %% Proposer Conditioning
```

**5. Functional Requirements**

The system must perform the following functions, referencing the AZR paper sections:

| Req ID | Requirement Description | AZR Paper Ref |
| :----- | :---------------------- | :------------ |
| FR01   | Load and initialize the Qwen2.5-Coder-3B model. | Sec 4.1 |
| FR02   | Implement the Proposer role using the LLM to generate task proposals based on task type and K historical references from the buffer. | Sec 3.1, 3.2, 3.3.2, Figs 34-36 |
| FR03   | Implement the Solver role using the LLM to generate solutions (code, input, or output) for given tasks. | Sec 3.1, 3.2, Figs 37-39 |
| FR04   | Implement Deduction task generation (propose `p, i`) and solving (predict `o` given `p, i`). | Sec 3.2 |
| FR05   | Implement Abduction task generation (propose `p, i`) and solving (predict `i` given `p, o`). | Sec 3.2 |
| FR06   | Implement Induction task generation (sample `p`, propose `{i^n}, m`) and solving (predict `p` given `m` and subset of `{(i^n, o^n)}`). | Sec 3.2 |
| FR07   | Initialize the Task Buffer with seed data (potentially using the base model or the single "identity" triplet). | Sec 3.3.1, Fig 5 |
| FR08   | Persistently store validated tasks (triplets, induction data) in the Task Buffer. | Sec 3.3.2 |
| FR09   | Sample K reference tasks uniformly from the appropriate buffer for proposer conditioning. | Sec 3.3.2 |
| FR10   | Fill partial batches for the solver by sampling from the buffer if proposer output is insufficient. | Sec 3.3.2 |
| FR11   | Implement Secure Code Executor: Validate proposed programs for Python syntax correctness. | Sec 3.3.3 (Step 1) |
| FR12   | Implement Secure Code Executor: Validate proposed programs against a list of forbidden modules (Fig 8). | Sec 3.3.3 (Step 2) |
| FR13   | Implement Secure Code Executor: Validate proposed programs for determinism by executing twice and comparing outputs. | Sec 3.3.3 (Step 3), Eq 7, Fig 13 |
| FR14   | Implement Secure Code Executor: Execute validated programs `p(i)` to generate ground truth outputs `o`. | Sec 3.2, 3.3.3 |
| FR15   | Implement Secure Code Executor: Verify solver's Abduction output `i_π` by checking `p(i_π) == o`. | Sec 3.3.4, Fig 10 |
| FR16   | Implement Secure Code Executor: Verify solver's Deduction output `o_π` by checking `o_π == o`. | Sec 3.3.4, Fig 11 |
| FR17   | Implement Secure Code Executor: Verify solver's Induction output `p_π` by checking `p_π(i^n) == o^n` for all held-out examples. | Sec 3.3.4, Fig 12 |
| FR18   | Calculate Learnability Reward (`r_propose`) based on solver's success rate on N rollouts of a proposed task. | Sec 3.1, Eq 4 |
| FR19   | Calculate Accuracy Reward (`r_solve`) based on binary correctness from verification (FR15-17). | Sec 3.1, Eq 5 |
| FR20   | Calculate final reward using the format-aware penalty structure. | Sec 3.1, Eq 6, Fig 33 |
| FR21   | Implement the TRR++ reinforcement learning update step (or PPO as fallback) using calculated rewards to update LLM weights. | Sec 3.3.5, Eq 8, Appendix A |
| FR22   | Implement the main orchestration loop as defined in Algorithm 1. | Algorithm 1 |
| FR23   | Log key metrics during the training loop (rewards, token lengths, buffer stats, etc.). | Sec C.3, C.4 |

**6. Non-Functional Requirements**

* **NFR01 (Performance):** The end-to-end AZ loop iteration time on a single A40 should allow for meaningful progress within reasonable timeframes (target < 1 hour per iteration initially, optimize as needed). Code execution within the sandbox should have a strict timeout (e.g., 10-15 seconds, per paper).
* **NFR02 (Scalability):** The initial implementation targets a single A40 node. Code structure (e.g., executor as a service endpoint, buffer access) should facilitate potential future scaling to multiple GPUs if needed, but multi-node orchestration is out of scope initially.
* **NFR03 (Reliability):** The main orchestration loop should handle expected errors (e.g., code execution failures, invalid task proposals, transient network issues) gracefully, log them, and continue operation where possible. Checkpointing of the LLM and buffer state should occur regularly.
* **NFR04 (Security):** The Secure Code Executor must effectively isolate executed code, preventing access to the host system, network (unless explicitly required and filtered), and sensitive data. Adherence to FR12 (forbidden modules) is critical.
* **NFR05 (Maintainability):** Code should be modular (separate components for orchestration, LLM interaction, executor client, buffer management, RL logic), well-commented, and use dependency management (e.g., `requirements.txt` or `conda environment.yml`).
* **NFR06 (Monitoring):** Implement detailed logging for all major steps and decisions within the loop. Track metrics specified in FR23 using standard logging and potentially tools like Weights & Biases for visualization.

**7. Infrastructure & Environment (RunPod)**

* **Compute Instance:** RunPod GPU Pod.
    * **GPU:** 1 x NVIDIA A40 (48GB GDDR6 VRAM).
    * **Instance Type:** Select a RunPod instance type providing the A40 along with adequate CPU/RAM (e.g., >= 9 vCPU, >= 50 GB System RAM).
* **Storage:**
    * **Network Volume:** Configure a RunPod Network Volume (e.g., 100-200 GB initial size, adjust as needed) mounted to the pod. Use for:
        * Task Buffer persistence.
        * LLM Checkpoint saving.
        * Persistent logs.
    * **Pod Ephemeral Storage:** Use for temporary files, code execution context.
* **Base Image:** Standard RunPod PyTorch image (or similar base image with CUDA drivers, Python, and standard ML libraries pre-installed). Example: `runpod/pytorch:2.3.0-py3.10-cuda12.1.1-devel-ubuntu22.04`.
* **Software Dependencies:**
    * Python 3.10+
    * PyTorch 2.x
    * Hugging Face `transformers`, `accelerate`, `datasets`
    * `trl` (for PPO/RL training)
    * `wandb` (optional, for logging)
    * Docker (if using Docker-based sandbox)
    * Libraries for code parsing/analysis (e.g., `ast`)
    * Specific libraries required by generated code (to be installed within the sandbox).

**8. Secure Code Execution**

* **Recommended Approach:** Docker-based Sandbox within the RunPod Pod.
    * **Implementation:**
        * Create a minimal Dockerfile for the executor environment, including Python and essential, safe libraries.
        * Use the Docker Python SDK or subprocess calls from the main Orchestrator script to:
            * Build/pull the executor image.
            * Run code within a new container for each execution request.
            * Mount necessary input files/code snippets into the container.
            * Capture stdout, stderr, and return codes from the container.
            * Apply strict resource limits (`--memory`, `--cpus`, potentially `ulimit` inside container).
            * Apply a timeout to the container execution.
            * Disable networking within the container (`--network none`) unless absolutely required for a specific, safe task.
            * Run the container as a non-root user.
    * **Filtering:** Before execution, parse the proposed code using Python's `ast` module or similar to explicitly disallow imports matching the forbidden list (Fig 8 in paper). Reject code containing disallowed imports before attempting execution.
* **Alternative (E2B):** If Docker setup proves too complex or insecure, investigate using the E2B service via its API. This requires managing E2B API keys securely.

**9. Configuration & Hyperparameters**

Configure the training loop using parameters derived from Table 3 in the AZR paper, adjusted for the 3B model and A40 GPU:

* **Model:** `Qwen/Qwen2.5-Coder-3B`
* **Max Prompt Length:** 6144 (or adjust based on Qwen3 capability/memory)
* **Max Response Length:** 8096 (or adjust)
* **RL Algorithm:** TRR++ (or PPO)
* **Optimizer:** AdamW
* **Learning Rate:** `1e-6` (or adjust)
* **Batch Size (Total):** Start smaller, e.g., 32 or 64 total examples across roles/tasks, adjust based on A40 48GB VRAM capacity during RL updates. The paper used `64*6=384` which is likely too large for a single A40.
* **PPO Epochs:** 1
* **KL Settings:** False (per paper)
* **Entropy Coeff:** 0.001
* **Rollout Temp:** 1.0
* **Rollout Top-P:** 1.0
* **K References (Buffer Sampling):** 6
* **N Samples (Task Accuracy Estimate):** 8
* **Format Penalty Values:** -0.5 (wrong but formatted), -1.0 (formatting error)
* **Seed Batch Factor:** 4
* **Max Programs (Buffer Size):** 16384 (or adjust based on storage/sampling needs)
* **Code Executor Timeout:** 10-15 seconds

**10. Security Plan**

* **Code Execution Sandbox:** Primary focus. Implement rigorous isolation, resource limits, timeouts, and import filtering as described in Section 8. Regularly review and update the forbidden module list.
* **RunPod Instance Security:** Secure access using strong SSH keys or RunPod credentials. Minimize exposed ports.
* **API Keys:** If using external services (like E2B or WandB), store API keys securely using environment variables or RunPod secrets management, not hardcoded.
* **Dependency Management:** Regularly scan dependencies for vulnerabilities.

**11. Monitoring & Logging Plan**

* **Logging:** Implement structured logging (e.g., using Python's `logging` module) for all components. Log key events like:
    * Loop iteration start/end.
    * Task proposal (type, references used).
    * Task validation results (pass/fail, reason).
    * Code execution results (success/failure, output/error, duration).
    * Solution verification results (correct/incorrect).
    * Calculated rewards (propose, solve, final).
    * RL update step details (loss, gradients if possible).
    * Buffer operations (additions, sampling).
    * Errors and exceptions.
* **Metrics Tracking (e.g., log to console/file and/or WandB):**
    * Average rewards per role/task type over time.
    * Average token lengths per role/task type over time.
    * Task proposal acceptance rate (ratio of valid tasks to proposed tasks).
    * Task buffer size and composition over time.
    * Code execution success/error rates and average duration.
    * RL training loss.
    * GPU Utilization, VRAM Usage, System Resource Usage (CPU/RAM).
    * (Optional) Task complexity/diversity metrics (AST edit distance, ComplexiPy score - Sec C.4).

**12. Milestones & Phases (Suggested)**

1.  **Phase 1: Environment Setup & Baseline (Est. 1-2 days):**
    * Provision RunPod A40 instance and Network Volume.
    * Set up base Docker image with dependencies.
    * Load Qwen2.5-Coder-3B model, perform basic inference tests.
2.  **Phase 2: Secure Code Executor Implementation (Est. 3-5 days):**
    * Implement the chosen sandboxing approach (Docker recommended).
    * Implement validation logic (syntax, safety, determinism).
    * Implement execution logic with resource limits and timeouts.
    * Thoroughly test with safe and malicious code examples.
3.  **Phase 3: Core AZR Components - No RL (Est. 4-6 days):**
    * Implement Proposer and Solver role logic (prompting, inference).
    * Implement the 3 task types generation structure.
    * Implement Task Buffer logic (in-memory or basic file storage initially).
    * Integrate basic task proposal -> validation -> solving -> verification flow using the executor.
4.  **Phase 4: Reward & RL Implementation (Est. 3-5 days):**
    * Implement reward calculation functions (learnability, accuracy, format penalty).
    * Integrate TRR++/PPO update using a library like `trl`.
    * Test RL update step with dummy rewards/data.
5.  **Phase 5: Integration & End-to-End Loop (Est. 2-4 days):**
    * Connect all components into the full Algorithm 1 loop.
    * Implement persistent Task Buffer using Network Volume.
    * Run initial short end-to-end tests (a few iterations).
    * Implement basic logging and monitoring.
6.  **Phase 6: Stabilization & Experimentation (Ongoing):**
    * Debug and stabilize the loop for longer runs (hours/days).
    * Refine monitoring and logging.
    * Run longer experiments, observe metrics, potentially tune hyperparameters.

**13. Risks & Mitigation**

* **Risk: RL Instability:** Training diverges or fails to improve.
    * **Mitigation:** Start with established PPO implementations (`trl`), carefully tune learning rate/batch size, monitor reward curves closely, ensure reward scaling is appropriate, potentially simplify reward function initially.
* **Risk: Code Executor Security Vulnerability:** Sandbox escape allows malicious code execution.
    * **Mitigation:** Rigorous implementation and testing of the sandbox (Section 8), strict filtering, minimal container privileges, regular security reviews.
* **Risk: Slow Iteration Time:** The loop takes too long, hindering experimentation.
    * **Mitigation:** Profile components (inference, execution, update), optimize bottlenecks, consider potential future scaling to multiple GPUs if RL update is the slowest part.
* **Risk: VRAM Limitations (Even with 48GB):** Very long contexts or large batches in RL updates exceed memory.
    * **Mitigation:** Use gradient accumulation, mixed-precision training (BF16), potentially model parallelism techniques if absolutely necessary (though complex for single node), reduce batch size/sequence length.
* **Risk: Cost Overruns:** Extended debugging or long runs consume budget.
    * **Mitigation:** Utilize RunPod's cheaper pricing, implement robust checkpointing, start/stop instance as needed, set up billing alerts.
* **Risk: Task Collapse/Lack of Diversity:** Proposer generates repetitive or trivial tasks.
    * **Mitigation:** Ensure K reference sampling works, monitor task diversity metrics (Sec C.4), potentially experiment with diversity-promoting rewards later if needed (Sec D.4).
