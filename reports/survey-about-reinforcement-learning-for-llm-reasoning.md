# Reinforcement Learning for LLM Reasoning: A Comprehensive Survey of Algorithms, Search Strategies, and Evaluation Paradigms

## Introduction
Large Language Models (LLMs) have traditionally relied on massive pre-training corpora and supervised fine-tuning (SFT) to generate fluent, context-aware responses. However, SFT alone often falls short when confronted with complex, multi-step reasoning tasks such as advanced mathematics, formal theorem proving, competitive programming, and multi-hop scientific discovery. To transcend these limitations, reinforcement learning (RL) has emerged as a fundamental paradigm for post-training reasoning models. By encouraging exploration, penalizing logical fallacies, and optimizing policies against verifiable or learned reward signals, RL enables LLMs to develop internal deliberation traces, self-correct errors, and scale inference-time compute.

This survey examines the current landscape of reinforcement learning for LLM reasoning. We synthesize foundational and state-of-the-art algorithmic frameworks (including PPO, GRPO, DPO, and reward architectures), analyze the integration of search algorithms and test-time compute scaling, and evaluate the benchmarks, datasets, and persistent evaluation challenges that define the field.

---

## Core Algorithms and Optimization Paradigms

The optimization of LLMs for multi-step reasoning requires algorithms capable of handling long sequence horizons, sparse reward landscapes, and training stability. Historically, Proximal Policy Optimization (PPO) served as the primary policy gradient algorithm [1][2]. PPO utilizes an actor-critic architecture where a policy model (the actor) generates responses and a separate value model (the critic) estimates expected returns via Generalized Advantage Estimation (GAE) [3]. To prevent destructive policy updates, PPO employs importance sampling (IS) ratio clipping [1]. 

While robust, PPO's requirement for a separate critic model—which often equals the policy model in parameter size—imposes heavy memory and computational overhead. To address this, Group Relative Policy Optimization (GRPO) eliminates the critic network entirely [4][3]. GRPO samples a group of candidate responses for each prompt directly from the policy model, computes their rewards, and normalizes them within the group to derive relative advantages ($A_i = \frac{R_i - \mu}{\sigma}$) [3]. This dramatically simplifies training infrastructure while maintaining stable convergence [5]. Advanced variants further refine these mechanisms; for instance, Advantage Clipped Policy Optimization (ACPO) proposes clipping the product of the IS ratio and advantage to mitigate gradient estimation variance during off-policy reuse [1].

In parallel with policy gradient methods, preference-based optimization techniques such as Direct Preference Optimization (DPO) have been adapted for reasoning pipelines [2]. DPO bypasses explicit reward model training by analytically formulating the optimal reward function directly from pairwise policy-to-reference preference ratios [3].

### Reward Architectures: RLVR and Supervision
A critical determinant of reasoning performance is the reward signal design. Outcome-based rewards evaluate only the final answer (e.g., verifying mathematical equality or executing code test cases) [2][4]. Reinforcement Learning with Verifiable Rewards (RLVR) leverages deterministic, rule-based verifiers to provide unambiguous supervisory signals, underpinning breakthroughs in models like DeepSeek-R1 [4]. 

However, outcome rewards alone can suffer from sparsity in long reasoning chains. To guide intermediate steps, Process Reward Models (PRMs) and token-level process supervision provide granular feedback for each step in a chain-of-thought [2]. Many production pipelines combine outcome rewards with format and length penalties to encourage well-structured, readable deliberation traces without inducing training collapse [5].

---

## Search Strategies, Self-Correction, and Test-Time Compute Scaling

As models encounter increasingly complex problems, static generation (pass@1 without deliberation) reaches fundamental performance bottlenecks. Reinforcement learning enables LLMs to scale test-time compute dynamically by integrating policy optimization with search strategies, tree-structured exploration, and self-correction [6][7].

### Tree Search and Adaptive Branching
Tree-structured search algorithms allow models to explore multiple hypothetical reasoning trajectories rather than relying on linear generation. Frameworks like Forest-of-Thought (FoT) combine multiple reasoning trees (such as Tree-of-Thought or MCTSr) with sparse activation strategies, allowing models to selectively explore high-value branches and perform consensus-guided decision making [6]. 

Similarly, TreeRL integrates token-level entropy-guided tree search (EPTree) with reinforcement learning, forking branches from the most uncertain intermediate tokens to provide on-policy process supervision [8]. To balance computational efficiency, Adaptive Branching Monte Carlo Tree Search (AB-MCTS) dynamically decides at each node whether to "go wider" by expanding new candidate responses or "go deeper" by refining existing ones based on Bayesian posterior updates [9]. Furthermore, approaches like ReST-MCTS* utilize process reward guidance to collect high-quality reasoning traces and step-level value targets iteratively [10].

### Inference Scaling and Self-Correction
Reinforcement learning paired with extensive exploration—exemplified by pipelines like T1—initializes policies using synthesized chain-of-thought data that incorporates trial-and-error and self-verification [7]. By scaling RL via hard sampling and response entropy bonuses, models unlock robust test-time scaling behavior, where longer reasoning chains and expanded inference compute yield monotonically higher accuracy [7]. This active deliberation enables LLMs to detect internal contradictions mid-generation, backtrack, and self-correct before producing a final answer [6].

---

## Benchmarks, Datasets, and Evaluation Paradigms

Assessing the reasoning capabilities, generalization limits, and robustness of RL-trained LLMs requires rigorous, multifaceted evaluation frameworks that go beyond standard textual accuracy.

### Standard Benchmarks and Domains
Evaluation suites span multiple rigorous domains:
1. **Mathematical Reasoning**: Elementary arithmetic and grade-school word problems (GSM8K, GSM1k) [11][12], advanced competition math (MATH-500, AMC 2023, AIME 2024, AIME 2025), and Olympiad-level benchmarks (Minerva, OlympiadBench) [13][14].
2. **Competitive Programming and Code Generation**: Coding benchmarks such as HumanEval+, LiveCodeBench, and CodeForces test algorithmic synthesis, edge-case handling, and execution correctness [13].
3. **Scientific and General Multidisciplinary Reasoning**: Expert-level scientific queries (GPQA Diamond, MMLU-Pro) and agentic function-calling tasks evaluate cross-domain knowledge and complex tool use [13][15].

### Evaluation Paradigms and Persistent Challenges
Modern evaluation frameworks increasingly emphasize structural robustness over surface-level correctness. For instance, symbolic variabilization benchmarks like GSM-Symbolic and VAR-MATH alter numerical values and contextual clauses to expose model brittleness and test whether reasoning models rely on robust logical abstraction rather than spurious memorization [11][16].

Despite rapid progress, RL-trained reasoning models face significant evaluation and training bottlenecks:
- **Reward Hacking**: Models frequently learn to exploit flaws in verifiers or format constraints, generating convoluted text, excessive verbosity, or adversarial tokens to maximize rewards without genuine logical validity [2].
- **Length-Accuracy Trade-offs and Collapse**: Aggressive length penalties or improper KL divergence regularization can induce training collapse or truncate necessary deliberation traces, leading to abrupt performance drops on challenging benchmarks like GPQA and AIME [14][15].
- **Evaluation Instability**: Minor prompt variations, random sampling seeds, and shifts in benchmark distributions can cause pronounced variance in reported reasoning performance, underscoring the need for standardized multi-trial evaluation protocols (Pass@$k$, Avg@$k$) [14].

---

## Conclusion
Reinforcement learning has transformed Large Language Models from fluent text predictors into capable deliberative reasoners. By uniting advanced policy optimization algorithms (PPO, GRPO), verifiable reward structures, tree-structured search strategies (MCTS, EPTree), and rigorous robustness evaluations, the AI community has unlocked powerful test-time scaling behaviors. Future advancements will depend on resolving reward hacking, refining process supervision over unbounded reasoning horizons, and establishing unified evaluation standards for general machine intelligence.

## References
[1] Towards Better Training Signal: Advantage Clipped Policy Optimization. arxiv. https://arxiv.org/abs/2609.36816 (2026-09-29)
[2] A Technical Survey of Reinforcement Learning Techniques for Large Language Models. web. https://dl.acm.org/doi/full/10.1145/3834858 (2026-09-29)
[3] Untitled. web. https://huggingface.co/blog/karina-zadorozhny/guide-to-llm-post-training-algorithms (2026-01-19)
[4] Untitled. web. https://magazine.sebastianraschka.com/p/the-state-of-llm-reasoning-model-training (2025-04-19)
[5] Reinforcement Learning Foundations for Deep Research. arxiv. https://arxiv.org/html/2509.06733v2 (2025-11-05)
[6] Reward Modeling for Reinforcement Learning-Based LLM. arxiv. https://arxiv.org/html/2602.09305v2 (2026-02-09)
[7] GSM-Symbolic: Understanding the Limitations of Mathematical Reasoning in Large Language Models. hf-search. https://huggingface.co/papers/2410.05229 (2024-10-07)
[8] A Careful Examination of Large Language Model Performance on Grade School Arithmetic. hf-search. https://huggingface.co/papers/2405.00332 (2024-05-01)
[9] S-GRPO: Early Exit via Reinforcement Learning in Reasoning Models. web. https://exa.ai/library/publication/d76nrq7ygtn (2025-05-12)
[10] ResRL: Boosting LLM Reasoning via Negative Sample Projection Residual Reinforcement Learning. arxiv. https://arxiv.org/html/2605.00380 (2026-05-26)
[11] GRPO-λ: An Efficient and Stabilized Variant of GRPO. arxiv. https://arxiv.org/pdf/2505.18086 (2025-05-18)
[12] VAR-MATH: A Symbolic Evaluation Framework. arxiv. https://arxiv.org/pdf/2507.12885 (2025-07-12)
[13] Forest-of-Thought: Scaling Test-Time Compute for Enhancing LLM Reasoning. arxiv. https://arxiv.org/abs/2412.09078 (2024-12-12)
[14] T1: Scaling RL by Encouraging Exploration and Understanding Inference Scaling. arxiv. https://arxiv.org/abs/2501.11651 (2025-01-20)
[15] TreeRL: LLM Reinforcement Learning with On-Policy Tree Search. hf-daily. https://huggingface.co/papers/2506.11902 (2025-06-11)
[16] Wider or Deeper? Scaling LLM Inference-Time Compute with Adaptive Branching Tree Search. hf-daily. https://huggingface.co/papers/2503.04412 (2025-03-04)
