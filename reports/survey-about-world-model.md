# A Comprehensive Survey on World Models in Artificial Intelligence and Reinforcement Learning: Architectures, Scaled Video Simulation, Evaluation, and Open Challenges

## Introduction

World models in artificial intelligence represent an internal generative simulation of an agent's environment, capturing spatial, temporal, and dynamic properties to forecast future states and evaluate decisions [1][2]. By shifting decision-making from reactive model-free paradigms to model-based latent imagination, world models enable sample-efficient policy learning, interactive environment simulation, and counterfactual physical reasoning [3][4]. 

This survey examines foundational architectures, large-scale video generation models functioning as world simulators (such as Sora and the Genie series), evaluation benchmarks (WorldLens, MiraBench, WorldBench), primary applications in autonomous driving and robotics, and persistent failure modes including kinematic rollouts and physical hallucinations.

---

## Foundational Architectures and Theoretical Paradigms

### The Vision-Memory-Controller ($V$-$M$-$C$) Framework
The modern era of deep world models was galvanized by Ha and Schmidhuber (2018), who introduced a compact tripartite architecture combining perception, temporal dynamics, and control [1]:
1. **Vision Model ($V$):** A Variational Autoencoder (VAE) that compresses high-dimensional pixel frames $x_t$ into low-dimensional latent vectors $z_t \in \mathbb{R}^{N_z}$.
2. **Memory Model ($M$):** A Recurrent Neural Network coupled with a Mixture Density Network (MDN-RNN) predicting the probability distribution of subsequent latent states conditioned on actions: $P(z_{t+1} \mid a_t, z_t, h_t)$.
3. **Controller ($C$):** A linear policy mapping compressed representations $[z_t, h_t]$ to actions $a_t$, optimized via Evolution Strategies.

By generating "hallucinated dreams" entirely in latent space using temperature-controlled sampling ($\tau$), policies can be trained efficiently inside imagination before physical deployment [1].

### Latent Dynamics Models and the Dreamer Series
To mitigate the accumulation of visual reconstruction errors in pixel space, subsequent research focused on latent dynamics models [3][5]. The Recurrent State-Space Model (RSSM), standardized across the Dreamer series, integrates deterministic recurrent states $h_t$ with stochastic representations $z_t$ [4]:
- **DreamerV1 (2019):** Solved continuous visual control tasks purely through multi-step latent imagination rollouts using analytic gradients [3].
- **DreamerV2 (2020):** Introduced discrete categorical RSSM representations and discrete action spaces, achieving human-level Atari benchmark performance [5].
- **DreamerV3 (2023–2025):** Established a unified reinforcement learning algorithm capable of solving over 150 diverse tasks with fixed hyperparameters—notably mining diamonds in Minecraft from scratch without human supervision—utilizing symlog transformations, KL balancing, and return normalization [2][4].

---

## Large-Scale Video Generation & Interactive World Simulators

### Sora-like Spacetime Diffusion Architectures
Scaling laws have transformed video generation models into powerful implicit world simulators [6]. Sora combines Diffusion Transformers (DiT) with spacetime latent patch tokenization, enabling training on variable resolutions, aspect ratios, and durations [6]. This large-scale scaling yields emergent physical reasoning capabilities, including 3D spatial consistency, object permanence under occlusion, and long-range temporal coherence [6]. Open-source implementations such as Open-Sora 2.0 and HunyuanVideo have democratized these capabilities using efficient Spatial-Temporal Diffusion Transformers (ST-DiTs) [7].

### Interactive World Simulators (Genie Series)
Moving beyond passive video generation, interactive world models such as Genie, Genie 2, and Genie 3 generate navigable 720p environments at 24 frames per second from text prompts or initial images [8]. Maintaining visual memory for up to a minute of continuous interaction, these models provide unlimited, exploration-driven simulation curriculums for training generalist embodied agents [8].

### Diffusion World Models for Reinforcement Learning (DIAMOND)
Integrating diffusion models directly into reinforcement learning loops, DIAMOND demonstrates that latent diffusion dynamics combined with policy optimization achieve state-of-the-art sample efficiency on complex interactive tasks like Atari 100k [9]. Similarly, cross-embodiment robot world models like DreamTrue leverage counterfactual post-training and action-conditioned multi-view prediction to ensure physical plausibility [10].

---

## Evaluation Benchmarks and Methodologies

Evaluating world models has transitioned from simple pixel-level metrics (PSNR, SSIM, Fréchet Video Distance) toward robust, action-conditioned, and physics-disentangled benchmarks:
- **Driving Benchmarks:** *WorldLens* evaluates driving world models across generation, reconstruction, action-following, and closed-loop navigation (using Predictive Driver Model Score and Arena Driving Score) [11]. *DrivingGen* and *ACT-Bench* assess visual realism, temporal coherence, and action fidelity [12][13], while *NuPlan* provides reactive multi-city closed-loop planning validation [14].
- **Robotic & Embodied Benchmarks:** *MiraBench* introduces hierarchical evaluation for action reliability, physical consistency, and optimism bias [15]. *WorldBench* evaluates concept-specific physical understanding (object permanence, friction coefficients, gravity acceleration, fluid viscosity), revealing that models often generate visually plausible trajectories while failing fundamental physical constants [16]. Additional frameworks like *WorldArena 2.0*, *World Reasoning Arena*, and *WorldModelBench* measure simulative reasoning and instruction-following fidelity [17][18][19].

---

## Primary Applications

### Autonomous Driving
World models serve as the backbone of modern autonomous driving architectures by ingesting multimodal sensor streams (cameras, LiDAR, HD maps) to construct 4D occupancy grids [20][21]. They enable closed-loop simulation, trajectory forecasting, and robust policy planning without exposing physical fleets to safety hazards [20].

### Robotics and Embodied AI
In robotics, world models facilitate model-based reinforcement learning and latent imagination rollouts, allowing agents to anticipate outcomes and plan multi-step manipulation tasks prior to physical execution [20][22]. Scalable interactive simulators (such as UniSim, IRASim, and CtrlWorld) support zero-shot transfer across unstructured environments [22].

---

## Open Challenges and Failure Modes

### Compounding Errors and Kinematic Rollouts
Autoregressive rollouts suffer from long-horizon error accumulation. Recent diagnostic frameworks distinguish between kinematic and dynamic imagination, demonstrating via Imagined Kinematic-Consistency Error (iKCE) that world models extrapolate kinematics blindly without reflecting true physical dynamics (such as surface friction changes or gait collapse) [23].

### Physical Hallucinations
Hallucinations in world models manifest across three dimensions: perceptual reconstruction errors, action-marginalized transitions due to sparse failure data coverage, and scene-diverging physical violations (e.g., momentum or gravity breaks) [24]. These failures are predictable via tokenizer round-trip residuals and flow instability [24].

### Optimism Bias
Because training corpora are heavily skewed toward successful demonstrations, action-conditioned simulators exhibit *optimism bias*—generating plausible success trajectories even when conditioned on failure-inducing actions [15].

---

## Future Directions and Conclusion

Future research in world models focuses on data-centric active coverage expansion to mitigate hallucinations [24], physics-aware modular simulators enforcing explicit physical constants [16], unified multimodal instruction-aware architectures [22], and latency-optimized latent planning for real-time robotic control [20]. By bridging large-scale generative pre-training with rigorous physical grounding, world models continue to drive the evolution toward autonomous generalist AI agents.

## References
[1] World Models. arxiv. https://arxiv.org/abs/1803.10122 (2018-03)
[2] Mastering Diverse Domains through World Models. arxiv. https://arxiv.org/abs/2301.04104 (2023-01)
[3] Dream to Control: Learning Behaviors by Latent Imagination. arxiv. https://arxiv.org/abs/1912.01603 (2019-12)
[4] Mastering Diverse Domains through World Models (Nature). web. https://www.nature.com/articles/s41586-025-08744-2 (2025-02)
[5] Mastering Atari with Discrete World Models. arxiv. https://arxiv.org/abs/2010.02193 (2020-10)
[6] Video Generation Models as World Simulators. web. https://openai.com/index/video-generation-models-as-world-simulators/ (2024-02)
[7] Open-Sora 2.0: Democratizing Efficient Spatial-Temporal Diffusion Transformers. hf-search. https://huggingface.co/papers/2503.09642 (2025-03)
[8] Genie 3: A New Frontier for World Models. web. https://deepmind.google/blog/genie-3-a-new-frontier-for-world-models/ (2025-04)
[9] DIAMOND: Diffusion World Models as General RL Agents. hf-search. https://huggingface.co/papers/2405.12399 (2024-05)
[10] DreamTrue: Action-Faithful Multi-View Robot World Models. hf-search. https://huggingface.co/papers/2610.12468 (2026-10)
[11] WorldLens: Full-Spectrum Evaluation Benchmark for Driving World Models. web. https://worldbench.github.io/assets_common/papers/worldlens.pdf (2024-12)
[12] DrivingGen: Generative Video World Models in Autonomous Driving. hf-search. https://huggingface.co/papers/2601.01528 (2026-01)
[13] ACT-Bench: Assessing Action Fidelity in World Models. hf-search. https://huggingface.co/papers/2412.05337 (2024-12)
[14] NuPlan: A Closed-Loop ML Planning Benchmark for Autonomous Vehicles. hf-search. https://huggingface.co/papers/2106.11810 (2021-06)
[15] MiraBench: Evaluating Action-Conditioned Reliability in Robotic World Models. arxiv. https://arxiv.org/html/2605.29360 (2026-05)
[16] WorldBench: Disentangled Evaluation of Physical Understanding in Video Generation. arxiv. https://arxiv.org/html/2601.21282v1 (2026-01)
[17] WorldArena 2.0: Benchmarking Embodied World Models on Modality and Functionality. hf-search. https://huggingface.co/papers/2605.17912 (2026-05)
[18] World Reasoning Arena: Simulative Reasoning and Long-Horizon Forecasting. hf-search. https://huggingface.co/papers/2603.25887 (2026-03)
[19] WorldModelBench: Instruction-Following and Physics-Adherence Evaluation. hf-search. https://huggingface.co/papers/2502.20694 (2025-02)
[20] A Comprehensive Survey of World Models for Autonomous Driving and Embodied AI. web. https://dl.acm.org/doi/10.1145/3746449 (2025-01)
[21] Survey of World Models for Autonomous Driving. arxiv. https://arxiv.org/html/2501.11260v4 (2025-01)
[22] Comprehensive Survey on World Models for Embodied AI. web. https://www.arxiv.org/pdf/2510.16732 (2025-10)
[23] Imagined Rollouts Are Kinematic, Not Dynamic. arxiv. https://arxiv.org/abs/2607.05966 (2026-07)
[24] Hallucination in World Models is Predictable and Preventable. arxiv. https://arxiv.org/html/2606.27326 (2026-06)
