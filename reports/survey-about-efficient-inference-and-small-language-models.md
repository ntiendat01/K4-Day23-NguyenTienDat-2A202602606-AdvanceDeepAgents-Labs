# Survey on Efficient Inference and Small Language Models

## Introduction and Motivation

The rapid scaling of Large Language Models (LLMs) has unlocked unprecedented capabilities across natural language processing, reasoning, and code generation. However, this scaling trajectory comes with steep operational costs: massive memory footprints, intensive storage requirements, high latency, and prohibitive energy consumption during inference. To address these deployment barriers, the research community has increasingly focused on **Small Language Models (SLMs)** and **efficient inference techniques** [1][2]. SLMs bridge the gap between resource constraints and competitive task performance by optimizing model design, leveraging compression, and executing workloads on high-throughput serving systems [1][3][4]. This survey synthesizes recent advancements in SLM design principles, model-level inference optimization, hardware-software co-design, and production serving architectures.

---

## Small Language Model Design and Compression Strategies

Achieving high intelligence within a compact parameter budget requires deliberate architectural innovations and post-processing compression [1][3]. Unlike massive foundation models, SLMs are engineered from the ground up or derived from larger teacher models using targeted adaptation workflows [4][5].

### Architectural Optimization
Lightweight decoder-only architectures incorporate efficient attention mechanisms such as grouped-query attention and FlashAttention to drastically cut memory overhead and latency [1]. Parameter and embedding sharing strategies—exemplified by specialized architectures like MobileLLM—reallocate capacity toward depth and intermediate representations rather than bloated embedding layers, enabling superior linguistic retention under strict parameter bounds [1]. Furthermore, neural architecture search and structural building blocks are employed to build lean pre-training foundations [3].

### Model Pruning
Pruning eliminates redundant parameters or structural components to reduce model capacity and runtime footprint [1]. Techniques are broadly divided into unstructured and structured pruning [1]. Structured pruning targets entire components such as attention heads, neurons, channels, or layers (e.g., utilizing Block Influence metrics in ShortGPT or LLM-Pruner) to directly shrink tensor dimensionality and memory bandwidth pressure [4]. Iterative pruning combined with retraining restores accuracy and effectively compresses over-parameterized networks into agile student models [3].

### Knowledge Distillation
Knowledge distillation (KD) transfers the inductive biases, reasoning capabilities, and internal representations of a larger teacher model to an efficient student model [1][4]. White-box and black-box KD methodologies transfer teacher logits, internal hidden states, and probabilistic outputs, overcoming distribution gaps through specialized distillation losses and generalized Kullback-Leibler (KL) divergence [4]. Empirical studies show that strategic pruning combined with knowledge distillation yields compact models that rival or surpass models trained entirely from scratch on equivalent compute budgets [5].

---

## Model-Level Inference Optimization and Quantization

During inference, execution bottlenecks shift dynamically between computational limits and memory bandwidth limits [2][6]. The inference lifecycle is divided into two distinct operational phases: the **prefilling stage**, which is compute-bound and handled via General Matrix Multiplication (GEMM) where weight-activation quantization enables Tensor Core acceleration, and the **decoding stage**, which is memory-bound and handled via General Matrix-Vector Multiplication (GEMV) where weight-only quantization accelerates memory access [2].

### Weight-Only and Weight-Activation Quantization
To alleviate memory bandwidth bottlenecks, quantization reduces numerical precision from FP16 to INT8, INT4, or FP8 formats [2][6]. Prominent weight-only quantization techniques include:
- **GPTQ**: Uniform left-to-right row quantization utilizing second-order Hessian approximations to minimize quantization error [2].
- **AWQ (Activation-aware Weight Quantization)**: Protects salient weight channels corresponding to prominent activation outliers via dynamic reparameterization [2].
- **Outlier-Resilient Methods (OWQ, SpQR, SqueezeLLM)**: Allocate mixed-precision formats to stubborn outlier weights [2].
- **Rotation-Based Methods (QuaRot, SpinQuant)**: Apply randomized Hadamard or trained orthogonal rotations to eliminate activation outliers prior to quantization [2].

For weight-activation quantization, methods like SmoothQuant, OmniQuant, and QServe migrate quantization difficulty from activations to weights, enabling end-to-end low-precision execution [6].

### Memory and Storage I/O Optimization
Inference I/O is decomposed across three primary data movements: model weight I/O ($W$), Key-Value (KV) cache I/O ($K$), and activation I/O ($A$) [7]. FlashAttention and FlashInfer rewrite self-attention as tiled block-wise algorithms operating within SRAM cache, eliminating redundant $O(N^2)$ reads and writes to High Bandwidth Memory (HBM) [6][7]. 

### Speculative Decoding
To amortize the heavy cost of loading model weights during autoregressive token generation, speculative decoding employs a lightweight draft model (or parallel speculative heads) to propose multiple candidate tokens, which are then verified in a single parallel forward pass by the target model [6][7]. Advanced variants like SpecInfer, Sequoia, tree-structured speculative decoding, and EAGLE increase acceptance rates and arithmetic intensity, shifting the operational point on the compute roofline and substantially accelerating wall-clock generation speed [6][7].

---

## Hardware-Software Co-Design and Production Serving Frameworks

Translating algorithmic efficiency into real-world throughput and low latency requires robust systems engineering, specialized hardware acceleration, and dynamic memory management [6][8].

### Memory Management and Serving Engines
Modern serving frameworks—such as vLLM, TensorRT-LLM, and Text Generation Inference (TGI)—solve memory fragmentation and high KV-cache overhead through advanced abstractions [8][9][10]:
- **PagedAttention**: Manages KV-cache memory in non-contiguous physical blocks (akin to virtual memory paging in operating systems), virtually eliminating internal fragmentation and supporting massive concurrent batch sizes [8][9][10].
- **Continuous (In-Flight) Batching**: Dynamically inserts and retires sequence requests at the iteration level rather than waiting for rigid batch boundaries, maximizing GPU utilization [8][9][10].
- **Prefix Caching and Chunked Prefill**: Shares common prompt prefixes across multiple requests and chunks long prompts to balance prefill latency with generation throughput [8][10].

### Hardware Accelerators and Multi-Backend Ecosystems
Production deployments leverage heterogeneous hardware accelerators including NVIDIA GPUs, AMD GPUs, Intel Gaudi, Google TPUs, AWS Trainium/Inferentia (Neuron), and Apple Silicon [8][11]. Multi-backend runtimes like TGI provide unified execution across CUDA, TensorRT-LLM, llama.cpp, and Neuron backends, allowing operators to optimize performance and power efficiency across diverse cloud and edge environments [11]. Furthermore, empirical studies on LLM serving in the wild demonstrate that prefill-decode disaggregation architectures successfully isolate compute-bound prefill workloads from memory-bound decoding workloads, preventing head-of-line blocking and stabilizing Tail Latency (P99) under high request concurrency [6][9].

---

## Conclusion

The intersection of small language model design and efficient inference has transformed generative AI from a resource-prohibitive endeavor into a highly optimizable stack. By combining architectural innovations, pruning, and knowledge distillation, SLMs achieve strong task competence at a fraction of the parameter footprint [1][4]. When paired with advanced model quantization, I/O-aware attention kernels, speculative decoding, and paged serving engines, these models deliver exceptional throughput and low latency across heterogeneous hardware ecosystems [2][7][8]. Future work will continue to refine hardware-software co-design principles, pushing the boundaries of edge intelligence and cost-effective cloud-scale deployment [6].

## References
[1] A Survey of Small Language Models. arxiv. https://arxiv.org/abs/2410.20011 (2024-10-25)
[2] A Survey on Efficient Inference for Large Language Models. arxiv. https://arxiv.org/abs/2404.14294 (2024-04-22)
[3] A Survey on Small Language Models. web. https://aclanthology.org/2025.ranlp-1.93/ (2025-01-01)
[4] A Comprehensive Survey of Small Language Models in the Era of Large Language Models. web. https://dl.acm.org/doi/10.1145/3768165 (2025-11-24)
[5] Compact Language Models via Pruning and Knowledge Distillation. hf-search. https://huggingface.co/papers/2407.14679 (2024-07-19)
[6] Efficient LLM Inference: A Survey and Data-Movement Framework. hf-search. https://huggingface.co/papers/2603.14989 (2026-03-19)
[7] I/O for LLM Inference: A Survey of Storage and Memory Bottlenecks. web. https://exa.ai/library/publication/r4nlm2lc7t2 (2026-03-19)
[8] vllm-project/vllm: A high-throughput and memory-efficient inference and serving engine for LLMs. web. https://github.com/vllm-project/vllm/ (2023-02-09)
[9] NVIDIA TensorRT LLM. web. https://developer.nvidia.com/tensorrt-llm (N/A)
[10] Text Generation Inference (TGI) v3 overview. web. https://huggingface.co/docs/text-generation-inference/main/conceptual/chunking (N/A)
[11] Multi-backend support · Hugging Face. web. https://huggingface.co/docs/text-generation-inference/main/en/multi_backend_support (N/A)
