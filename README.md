🚀 CPU-Optimized Transformer Inference Engine in Pure C
=======================================================

> A research-oriented, CPU-first Transformer inference engine written entirely in ISO C, focused on low-level performance engineering, memory efficiency, and reproducible benchmarking.

This project is a systematic study of optimizing Transformer inference on CPUs without external ML libraries.It emphasizes cache efficiency, memory discipline, quantization, operator fusion, and tiled attention — all implemented from scratch.

🎯 Vision
---------

Modern LLM systems are GPU-centric.This project explores:

*   What happens when we treat **CPU as the primary target**
    
*   How far low-level systems engineering can push Transformer inference
    
*   How memory design influences real-world latency and throughput
    

The goal is to engineer a **reproducible, measurable optimization journey** — not just implement a model.

🏗 Architecture Overview
========================
```
Input → Embedding → Multi-Head Attention → FFN → LayerNorm → Output   `
```

*   Encoder-style Transformer
    
*   Inference-only execution path (inference\_forward)
    
*   No training-time artifacts
    
*   Deterministic memory allocation strategy
    
*   Benchmark-driven development
    

⚙️ Optimization Phases
======================

🔥 PHASE 1 — Brutal Low-Level Optimizations
===========================================

### 1️⃣ Quantization (Primary Lever)

*   Int8 weight quantization
    
*   Scaled integer matmul
    
*   Accuracy vs latency tradeoff measured
    

### 2️⃣ Operator Fusion

*   Linear + Bias fusion
    
*   LayerNorm + residual fusion
    
*   Eliminates redundant memory traffic
    
*   Reduces cache thrashing
    

### 3️⃣ Cache-Aware Tensor Layouts

*   Contiguous memory packing
    
*   Layout rearranged for CPU cache line efficiency
    
*   Stride elimination
    

### 4️⃣ Threading (Pure C)

*   Manual thread pool
    
*   Work partitioning across heads
    
*   No OpenMP / no external libs
    

### 5️⃣ Inference-Only Execution

Separate surgical forward path:
```
  void inference_forward(Model *model, Tensor *input);   `
```
Removed:

*   Gradient buffers
    
*   Training-only caches
    
*   Unused intermediate tensors
    

🧠 PHASE 2 — CPU-First Attention
================================

### 6️⃣ Flash-Attention-Like CPU Tiling

*   Blocked attention computation
    
*   No full attention matrix allocation
    
*   Softmax computed on-the-fly
    
*   Tile-based memory reuse
    

Designed specifically for CPU memory hierarchy.

### 7️⃣ Head Pruning + Structured Sparsity

*   Pruned low-impact attention heads
    
*   Structured sparsity for SIMD friendliness
    

### 8️⃣ KV Cache (Autoregressive Mode)

*   Appended token-level KV
    
*   Avoids recomputation for previous tokens
    
*   Enables efficient generation
    

🧬 PHASE 3 — Memory Engineering
===============================

Memory is treated as a first-class design constraint.

### 9️⃣ Arena Allocator

*   Zero malloc/free in hot path
    
*   Preallocated contiguous memory blocks
    
*   Deterministic allocation
    

### 🔟 Activation Reuse

*   Manual buffer reuse across layers
    
*   Reduced peak activation memory
    
*   Checkpoint-like strategy for inference
    

### 1️⃣1️⃣ Stack Allocation

*   Temporary buffers allocated on stack where safe
    
*   Eliminates heap pressure
    

📊 PHASE 4 — Benchmarking & Measurement
=======================================

All optimizations are versioned and benchmarked.

Metrics tracked:

*   ⏱ Latency (ms per sentence / token)
    
*   🚀 Throughput (tokens/sec)
    
*   🧠 Peak RSS memory
    
*   📉 Activation memory footprint
    
*   🎯 CoNLL entity-level F1 score
    

Example benchmark snapshot:


| Version | Latency (ms) | Tokens/sec | RSS (MB) | F1 |
| :--- | :--- | :--- | :--- | :--- |
| v0.0 | 14.2 | 712 | 142 | 91.2 |
| v0.2 | 11.1 | 890 | 96 | 91.2 |
| v0.5 | 8.9 | 1140 | 74 | 90.9 |

All results are reproducible with fixed compiler flags and dataset subsets.

📁 Project Structure
====================

/src
   attention.c    
   linear.c    
   quant.c    
   arena.c    
   tensor.c    
   model.c    
   inference.c
/benchmarks    
   run_benchmark.sh    
   results.csv
/docs    
   baseline.md    
   arena_allocator.md    
   quantization.md    
   tiled_attention.md   `

🛠 Build
========
```
   gcc -O3 -march=native -o transformer src/*.c -lm -lpthread   `
```
🧪 Run Benchmark
================
```
./transformer --benchmark   `
```
Outputs:

*   Tokens/sec
    
*   Latency per sentence
    
*   Peak memory usage
    
*   Git version tag
    

📌 Engineering Principles Followed
==================================

*   Deterministic memory
    
*   No dynamic allocation in hot path
    
*   Measure before optimize
    
*   Versioned performance evolution
    
*   CPU cache-aware design
    
*   Minimal abstraction overhead
    

🧠 Key Engineering Insights
===========================

*   Memory bandwidth often dominates arithmetic cost
    
*   Arena allocation drastically reduces allocator overhead
    
*   Attention tiling reduces peak memory and improves locality
    
*   Operator fusion is often more impactful than threading
    
*   Quantization provides best latency gains per complexity added
    

📚 Technical Deep Dives
=======================

Detailed design documents:

*   Arena Allocator Design
    
*   CPU-Tiled Attention Algorithm
    
*   Activation Memory Analysis
    
*   Quantization Tradeoff Study
    

🧪 Tested On
============

*   GCC with -O3 -march=native
    
*   Linux x86\_64
    
*   Intel and AMD CPUs
    

🎓 Why This Project Exists
==========================

This repository demonstrates:

*   Systems-level thinking applied to ML
    
*   Memory-first engineering
    
*   Performance-driven development
    
*   Ability to reason about CPU architecture
    
*   Reproducible optimization workflow
    

It is designed as a serious proof-of-work artifact for:

*   Systems ML Engineering
    
*   Inference Optimization
    
*   Memory-Efficient AI Systems
    
*   CPU-first deployment research
    

🏁 Roadmap
==========

*   SIMD intrinsics (AVX2/AVX512)
    
*   Weight-only 4-bit quantization
    
*   Sparse attention kernels
    
*   Microbenchmark suite
    
*   Cross-platform profiling
    

📄 License
==========

MIT License
