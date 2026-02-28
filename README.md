A CPU-optimized transformer inference engine written in pure C, 
focused on low-memory, low-latency inference via quantization, 
operator fusion, cache-aware layouts, and arena-based memory management.


Pure C (no BLAS, no frameworks)

INT8 / INT4 inference

CPU-first attention (flash-like tiling)

Arena allocator (zero malloc in hot path)

KV cache for autoregressive decoding

Reproducible benchmarks

Architecture:

Tokenizer
   ↓
Embedding
   ↓
[ Fused Transformer Block ] × N
   ↓
LM Head
