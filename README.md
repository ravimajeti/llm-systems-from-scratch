# llm-systems-from-scratch

A small Llama-style LLM (~114M params, GPT-2-small scale: RoPE, RMSNorm, SwiGLU, GQA, QK-norm, document-masked packing), trained
from scratch and then made fast **one measured bottleneck at a time**: input pipeline, numerics, Triton/CUDA
kernels, memory, distributed training, FP8, and later an inference engine benchmarked against vLLM.

Every optimization follows the same loop: profile, predict, change one thing, measure, check correctness against
the baseline, write it up. Results are reported in throughput, dollars, and joules.

**Status:** Week 1 (setup). See [PLAN.md](PLAN.md) for the method and phases.

## Quickstart

```bash
uv sync --extra cpu      # Mac or CPU (macOS wheels include MPS)
uv sync --extra cu128    # NVIDIA GPU machine
uv run pytest
```

## Layout

```text
src/llmsys/     library code (model.py: baseline model)
tests/          correctness tests (CPU-runnable; run in CI)
docs/designs/   design docs (start with baseline-model.md)
```

## License

MIT
