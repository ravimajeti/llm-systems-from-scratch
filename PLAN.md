# Plan

## Method

1. **Profile → hypothesize → change one thing → measure → verify correctness → write it up.** No optimization
   without profiler evidence that it targets a real bottleneck.
2. **Correctness guard:** every change reproduces the baseline loss curve (same seed, same data order) within
   tolerance. Kernels are tested against a PyTorch reference (forward + backward); parallel code against a
   single-GPU reference.
3. **Compare against references,** not only against our own previous version: torch.compile, Liger Kernel,
   FlashAttention, TorchTitan, Megatron-Core, TransformerEngine, vLLM. Build it ourselves first, then compare.
4. **Predict, then measure.** A performance model predicts each experiment before it runs; the gap is explained
   afterwards. Every kernel reports its roofline ceiling and the % of it reached.
5. **Measure noise:** repeated runs, reported variance. A win inside the noise band is not a win.
6. **Negative results get written up too.**
7. **Only architecture features real modern models use.**

## Model

- Llama-style decoder, ~124M params (12 layers, d=768): RoPE, RMSNorm, SwiGLU, no biases.
- Modern features: grouped-query attention, QK-norm, packed sequences with per-document attention masking.
- Later: a DeepSeek-style fine-grained MoE variant.
- Data: FineWeb-Edu, GPT-2 tokenizer.

## Metrics

Tokens/s, MFU, step-time breakdown, peak memory, time-to-target-loss; for inference TTFT, TPOT, throughput, and %
of bandwidth roofline. Everything is also reported as **dollars** and **joules** (GPU-only energy from hardware
counters).

## Phases

| Phase | Topic |
|---|---|
| 0 | Setup: data prep, naive baseline, profiling harness, correctness tests, CI |
| 1 | Input pipeline & OS: memmap data, loaders, pinned memory, clocks, power-limit study, fault-tolerant resume |
| 2 | Numerics & attention: bf16, TF32, fused optimizers, torch.compile, SDPA vs FlexAttention vs varlen FA |
| 3 | Kernels: Triton RMSNorm, fused QKV→QK-norm→RoPE, SwiGLU, fused linear + cross-entropy, CUDA graphs |
| 4 | Memory: snapshots, chunked loss, activation checkpointing |
| 5 | Distributed: DDP → FSDP2, 1→8 GPU scaling; tensor / pipeline / context parallelism |
| 6 | FP8, optimizer work, train-to-target milestone |
| 7 | MoE: routing, grouped GEMM, expert parallelism |
| 8 | Inference engine: KV cache, batching, paged attention, quantized decode kernels, vs vLLM |
| 9 | Portability: AMD MI300X, AWS Trainium / Inferentia |
