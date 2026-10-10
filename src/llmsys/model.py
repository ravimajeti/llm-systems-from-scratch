"""Baseline Llama-style decoder (Phase 0).

Deliberately naive: fp32, attention materializes the full T x T score matrix with an explicit mask,
GQA by copying K/V heads, separate Q/K/V projections, unpadded vocab, full-size logits for the loss.
Each of these is a later, measured optimization target. Architecture: docs/designs/baseline-model.md.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int = 50257  # GPT-2 tokenizer
    n_layers: int = 12
    d_model: int = 768
    n_heads: int = 12  # query heads
    n_kv_heads: int = 4  # GQA; set equal to n_heads for standard multi-head attention
    ffn_hidden: int = 2048  # SwiGLU hidden size, 8/3 * d_model
    max_seq_len: int = 1024
    rope_theta: float = 10000.0
    norm_eps: float = 1e-6
    qk_norm: bool = True
    doc_masking: bool = True
    eot_id: int = 50256  # <|endoftext|>; starts every document in the packed token stream
    tie_embeddings: bool = True
    init_std: float = 0.02

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        assert self.n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        assert 0 <= self.eot_id < self.vocab_size

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps) * self.weight


# ---- Rotary position embedding (rotate-half layout, as in Hugging Face Llama/Qwen checkpoints) ----


def rope_cos_sin(seq_len: int, head_dim: int, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    """cos/sin tables of shape (seq_len, head_dim). Pair i rotates at frequency theta^(-2i/head_dim)."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    angles = torch.outer(torch.arange(seq_len, dtype=torch.float32), inv_freq)  # (T, head_dim/2)
    angles = torch.cat([angles, angles], dim=-1)  # dims i and i + head_dim/2 share an angle
    return angles.cos(), angles.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: (B, heads, T, head_dim); cos/sin: (T, head_dim)."""
    return x * cos + rotate_half(x) * sin


# ---- Attention ----


def doc_ids_from_tokens(idx: torch.Tensor, eot_id: int) -> torch.Tensor:
    """Document index per position: increments at every <|endoftext|>, which starts a new document."""
    return (idx == eot_id).cumsum(dim=-1)


def build_attention_mask(idx: torch.Tensor, eot_id: int, doc_masking: bool) -> torch.Tensor:
    """Boolean mask, True = may attend. Shape (1, 1, T, T), or (B, 1, T, T) with doc masking."""
    T = idx.shape[1]
    mask = torch.ones(T, T, dtype=torch.bool, device=idx.device).tril()[None, None]
    if doc_masking:
        doc = doc_ids_from_tokens(idx, eot_id)
        same_doc = doc[:, :, None] == doc[:, None, :]  # (B, T, T)
        mask = mask & same_doc[:, None]
    return mask


def naive_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """q: (B, H, T, D); k, v: (B, H_kv, T, D); mask broadcastable to (B, H, T, T).

    Query head h reads K/V head h // (H / H_kv), matching Hugging Face's repeat_kv.
    """
    n_rep = q.shape[1] // k.shape[1]
    k = k.repeat_interleave(n_rep, dim=1)
    v = v.repeat_interleave(n_rep, dim=1)
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])  # (B, H, T, T)
    scores = scores.masked_fill(~mask, float("-inf"))  # every row keeps its diagonal, so no all -inf rows
    return scores.softmax(dim=-1) @ v


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads, self.n_kv_heads, self.head_dim = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
        self.q_proj = nn.Linear(cfg.d_model, cfg.n_heads * cfg.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.d_model, bias=False)
        # QK-norm (Qwen3 style): per-head RMSNorm over head_dim, one weight shared across heads, before RoPE.
        self.q_norm = RMSNorm(cfg.head_dim, cfg.norm_eps) if cfg.qk_norm else nn.Identity()
        self.k_norm = RMSNorm(cfg.head_dim, cfg.norm_eps) if cfg.qk_norm else nn.Identity()

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, mask: torch.Tensor):
        B, T, _ = x.shape
        q = self.q_norm(self.q_proj(x).view(B, T, self.n_heads, self.head_dim))
        k = self.k_norm(self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim))
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))  # (B, heads, T, head_dim)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        y = naive_attention(q, k, v, mask)
        return self.o_proj(y.transpose(1, 2).reshape(B, T, self.n_heads * self.head_dim))


class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.up_proj = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.down_proj = nn.Linear(cfg.ffn_hidden, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.mlp_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.mlp = SwiGLU(cfg)

    def forward(self, x, cos, sin, mask):
        x = x + self.attn(self.attn_norm(x), cos, sin, mask)
        return x + self.mlp(self.mlp_norm(x))


class Transformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight
        cos, sin = rope_cos_sin(cfg.max_seq_len, cfg.head_dim, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self._init_weights()

    def _init_weights(self):
        # GPT-2 style: N(0, 0.02) everywhere; projections that write into the residual stream are scaled by
        # 1/sqrt(2 * n_layers) so the residual's variance doesn't grow with depth. Norm weights stay at 1.
        std = self.cfg.init_std
        for name, p in self.named_parameters():
            if name.endswith(("o_proj.weight", "down_proj.weight")):
                nn.init.normal_(p, std=std / math.sqrt(2 * self.cfg.n_layers))
            elif p.dim() == 2:
                nn.init.normal_(p, std=std)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())  # tied weights are counted once

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        """idx, targets: (B, T) token ids. Returns (logits (B, T, vocab), loss or None)."""
        T = idx.shape[1]
        assert T <= self.cfg.max_seq_len, f"sequence length {T} > max_seq_len {self.cfg.max_seq_len}"
        mask = build_attention_mask(idx, self.cfg.eot_id, self.cfg.doc_masking)
        cos, sin = self.rope_cos[:T], self.rope_sin[:T]
        x = self.embed(idx)
        for layer in self.layers:
            x = layer(x, cos, sin, mask)
        logits = self.lm_head(self.norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss
