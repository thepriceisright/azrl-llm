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