# Comprehensive Survey on Video and Multimodal Generation

## Introduction

The landscape of generative artificial intelligence has experienced a profound paradigm shift, transitioning from traditional two-dimensional image synthesis and early GAN-based video models \[1] toward highly scalable Diffusion Transformers (DiTs), autoregressive (AR) sequence models, flow-matching formulations, and unified multimodal world simulators \[1]\[2]\[3]. Modern video generation models—such as Sora, CogVideoX, Wan, HunyuanVideo, and LTX-Video \[1]—are capable of generating high-resolution, physically consistent, and minute-long videos from textual prompts, image inputs, and multimodal instructions \[4]\[3]\[5].

This comprehensive survey explores the foundational architectures governing recent video generation \[1], the robust benchmarking suites and quantitative metrics used to evaluate temporal consistency and multimodal semantic alignment \[6]\[7]\[8], and the primary advances, long-video extrapolation strategies, and persistent challenges facing the field \[3]\[9]\[5].

---

## Foundational Architectures and Modeling Paradigms

Video generation models have evolved across several distinct modeling generations, moving away from UNet-based score-based generative models toward transformer backbones and hybrid formulations \[1]\[2]\[4].

### From GANs and VAEs to Diffusion Transformers (DiT)

Early video generation relied heavily on Generative Adversarial Networks (GANs) and Variational Autoencoders (VAEs), which struggled with scalability, mode collapse, and temporal coherence over extended horizons \[1]\[3]. The introduction of Denoising Diffusion Probabilistic Models (DDPMs) and Score-Based Generative Models (SGMs) established a robust probabilistic foundation \[10].

However, standard UNet backbones faced limitations in scaling capacity and capturing complex long-range dependencies. This motivated the adoption of Diffusion Transformers (DiTs) (e.g., W\.A.L.T, VDT, Latte, and FullDiT) \[2]\[10], which treat spatial and temporal tokens uniformly through self-attention. Advanced models like CogVideoX incorporate specialized architectural components, including 3D Causal VAEs for efficient spatio-temporal compression, Expert Adaptive LayerNorm (EALN), Multi-Resolution Frame Packing, and 3D Rotary Position Embeddings (3D-RoPE) \[4].

### Multimodal DiT (MM-DiT) and Flow Matching

To bridge natural language prompts with continuous video latents, modern architectures employ Multi-Modal Diffusion Transformer (MM-DiT) designs \[1]. MM-DiT frameworks typically utilize dual-stream modality-specific processing for text and video tokens in early layers, transitioning into single-stream cross-modal fusion layers where text embeddings directly modulate video generation via cross-attention \[1]. Furthermore, flow-matching formulations (such as Pyramidal Flow) integrate spatial and temporal pyramids with corrective renoising, improving sample efficiency and trajectory straightness compared to standard diffusion schedules \[4].

### Autoregressive and Hybrid AR-Diffusion Frameworks

While diffusion models excel at high-fidelity spatial details, autoregressive (AR) models and hybrid AR-diffusion frameworks offer distinct advantages for interactive streaming and real-world simulation \[1]\[2]. Models like CausVid re-engineer diffusion transformers into autoregressive architectures using causal block-wise attention, Key-Value (KV) caching, and Distribution Matching Distillation (DMD) to reduce sampling steps from \~50 down to 4, enabling real-time generation and interactive world modeling \[1]\[2].

---

## Evaluation Benchmarks, Datasets, and Metrics

Assessing the quality, temporal stability, and semantic alignment of generated videos has evolved significantly. Traditional full-reference metrics (such as Fréchet Video Distance \[FVD] and Video Inception Score) and per-frame CLIP scores often fail to capture temporal dynamics, open-domain variations, and human perceptual preferences \[8].

### Hierarchical Benchmark Suites: VBench and VBench++

To address these limitations, comprehensive benchmarking frameworks such as **VBench** and its expanded iteration **VBench++** evaluate video generative models across a hierarchical taxonomy of 16+ disentangled dimensions \[6]. These dimensions are divided into two coarse-grained pillars:

1. **Video Quality (Temporal & Frame-Wise)**:
   - *Subject Consistency*: Measures appearance stability across frames using DINO feature similarity \[6].
   - *Background Consistency*: Evaluates background stability using CLIP feature similarity \[6].
   - *Temporal Flickering*: Quantifies high-frequency local noise and flaws via mean absolute difference \[6].
   - *Motion Smoothness & Dynamic Degree*: Assesses physical motion realism using video interpolation priors and optical flow estimation via RAFT \[6].
   - *Aesthetic & Imaging Quality*: Evaluates frame-level visual appeal (LAION aesthetic predictor) and distortions/blur (MUSIQ predictor) \[6].
2. **Video-Condition Consistency (Semantics & Style)**:
   - Evaluates semantic alignment across object classes, multiple objects, human actions, colors, spatial relationships, and scenes using specialized vision models (e.g., GRiT, UMT, Tag2Text) \[6].

### Compositional Text-to-Video Benchmarking (T2V-CompBench)

To evaluate fine-grained compositional capabilities, **T2V-CompBench** tests models across 7 core categories (encompassing 1,400 prompts) including consistent attribute binding, dynamic attribute binding (e.g., color changes over time), spatial relationships, motion binding, action binding, object interactions, and generative numeracy (quantities from 1 to 8) \[7]. Evaluation pipelines utilize advanced Multimodal Large Language Models (MLLMs) such as Grid-LLaVA (processing uniform 6-image multi-frame grids with chain-of-thought prompting), frame-by-frame image LLMs (D-LLaVA), and detection-based spatial bounding box checks \[7].

### Human Judgment Datasets and Learned Metrics (T2VScore)

To align automated metrics with human perception, the **TVGE dataset** collects extensive human judgments across text-video alignment and video quality dimensions \[8]. The resulting **T2VScore** evaluator decouples evaluation into **T2VScore-A** (leveraging video LLMs and GPT-4V for prompt fidelity) and **T2VScore-Q** (predicting production-grade visual quality through expert models trained on generative video distortion patterns) \[8].

---

## Recent Advances, Long-Video Generation, and Applications

Recent advancements in text-to-video (T2V) and multimodal generation have unlocked sophisticated capabilities in long-form generation, controllable video editing, and world simulation \[3]\[9]\[5].

### Long-Video Generation Paradigms

Generating consistent videos beyond short clips (e.g., minutes rather than seconds) requires mitigating error accumulation and semantic drift \[3]\[9]\[5]. Existing methodologies follow two primary paradigms:

- **Divide-and-Conquer / Coarse-to-Fine Approaches**: Methods like NUWA-XL employ diffusion-over-diffusion hierarchies to synthesize coarse global structures before refining local frames \[3]. Similarly, StreamingT2V generates video in latent chunks (8-16 frames via VQ-GAN) using a Conditional Attention Module (CAM) for short-term consistency, an Appearance Preservation Module (APM) to prevent semantic drift, and randomized blending to extend videos beyond 1,200 frames (2+ minutes) at 720p resolution \[9].
- **Temporal Autoregressive & Memory Mechanisms**: Techniques such as CausVid, ART-V, LTVR, and Phenaki utilize causal attention, KV-caching, and retrospective memory mechanisms to maintain long-term subject and background consistency across extended sequences \[2]\[9]\[5].

### Controllable and Multimodal Generation

Beyond text-to-video, modern systems support rich conditioning modalities including image-to-video (I2V), video-to-video editing, camera trajectory control, bounding-box guidance, and audio synchronization \[9]. Training-free plug-and-play frameworks like UniCtrl utilize cross-frame self-attention and motion injection to preserve temporal consistency and eliminate flicker without modifying underlying foundational weights \[3].

---

## Current Challenges and Future Directions

Despite remarkable progress, several fundamental challenges remain in video and multimodal generation:

1. **Physical Law Adherence**: Generative models frequently violate real-world physical laws, exhibiting artifacts in gravity, object collisions, rigid-body dynamics, and occlusion handling \[3].
2. **Computational Complexity & Resource Scarcity**: Training and serving massive Diffusion Transformer and MLLM architectures demand immense GPU clusters, memory bandwidth (e.g., for KV-caching and long-context attention), and large-scale curated video datasets \[9]\[5].
3. **Semantic Drift & Temporal Artifacts**: Over long temporal horizons, maintaining consistent identity, fine-grained details, and flicker-free motion remains difficult \[3]\[9].
4. **Unified Multimodal Simulation**: Future research focuses on scaling unified world simulators capable of interactive, real-time reasoning across text, audio, images, and high-fidelity video streams \[1]\[3].

## References
[1] From Sora What We Can See: A Survey of Text-to-Video Generation. arxiv. https://arxiv.org/abs/2405.10674 (2024-05-17)
[2] Perception, Reason, Think, and Plan: A Survey on Large Multimodal Reasoning Models. arxiv. https://arxiv.org/abs/2505.04921 (2025-05-08)
[3] A Survey on Long Video Generation: Challenges, Methods.... hf-search. https://arxiv.org/html/2403.16407v1 (2024-03-24)
[4] Ming-Omni: A Unified Multimodal Model for Perception and Generation. hf-search. https://huggingface.co/papers/2506.09344 (2025-06-11)
[5] From Specific-MLLMs to Omni-MLLMs: A Survey on MLLMs Aligned with Multi-modalities. web. https://aclanthology.org/anthology-files/pdf/findings/2025.findings-acl.453.pdf (2025)
[6] The Dawn of Video Generation: Preliminary Explorations with SORA-like Models. arxiv. https://arxiv.org/abs/2410.05227 (2024-10-10)
[7] A survey on omni-modal language models. web. https://www.elspub.com/doi/10.55092/aiplus20260001 (2025-11-06)
[8] VBench++: Comprehensive and Versatile Benchmark Suite for Video Generative Models. hf-search. https://huggingface.co/papers/2411.13503 (2024-11-20)
[9] Generative AI Video Evaluation: Survey of Metrics, Benchmarks, and Trustworthiness. web. https://openaccess.thecvf.com/content/CVPR2026W/VGBE/papers/Safavigerdini_Generative_AI_Video_Evaluation_Survey_of_Metrics_Benchmarks_and_Trustworthiness_CVPRW_2026_paper.pdf (2026)
[10] Video Generation Models: A Survey of Post-Training and Alignment. hf-search. https://arxiv.org/html/2610.00812 (2026-09-30)
