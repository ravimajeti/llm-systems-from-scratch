import math
from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F

from llmsys.model import ModelConfig, Transformer, build_attention_mask, naive_attention

TINY = ModelConfig(
    vocab_size=128, n_layers=2, d_model=64, n_heads=4, n_kv_heads=2, ffn_hidden=128, max_seq_len=64, eot_id=0
)


def make_model(**overrides) -> Transformer:
    torch.manual_seed(0)
    return Transformer(replace(TINY, **overrides)).eval()


def random_tokens(*shape, low=1, high=TINY.vocab_size) -> torch.Tensor:
    """Tokens that avoid eot_id (0) unless placed explicitly."""
    return torch.randint(low, high, shape, generator=torch.Generator().manual_seed(1))


def test_param_count_default_config():
    with torch.device("meta"):
        model = Transformer(ModelConfig())
    cfg = model.cfg
    hd = cfg.head_dim
    per_layer = (
        cfg.d_model * hd * (cfg.n_heads + 2 * cfg.n_kv_heads)  # q, k, v projections
        + cfg.n_heads * hd * cfg.d_model  # o projection
        + 3 * cfg.d_model * cfg.ffn_hidden  # gate, up, down
        + 2 * hd  # q_norm, k_norm
        + 2 * cfg.d_model  # attn_norm, mlp_norm
    )
    expected = cfg.vocab_size * cfg.d_model + cfg.n_layers * per_layer + cfg.d_model  # tied embedding
    assert model.num_params() == expected == 114_115_584


def test_tied_embeddings():
    model = make_model()
    assert model.lm_head.weight is model.embed.weight
    untied = make_model(tie_embeddings=False)
    assert untied.num_params() == model.num_params() + TINY.vocab_size * TINY.d_model


def test_initial_loss_near_uniform():
    model = make_model()
    idx = random_tokens(4, 32)
    targets = torch.randint(0, TINY.vocab_size, (4, 32), generator=torch.Generator().manual_seed(2))
    # Random targets, not targets=idx: tied embeddings give each token's own logit a head start at init.
    _, loss = model(idx, targets)
    assert abs(loss.item() - math.log(TINY.vocab_size)) < 0.3


def test_causal():
    model = make_model()
    a = random_tokens(1, 32)
    b = a.clone()
    b[:, 20:] = a[:, 20:] % (TINY.vocab_size - 1) + 1  # changes every later token, never to eot_id
    la, _ = model(a)
    lb, _ = model(b)
    torch.testing.assert_close(la[:, :20], lb[:, :20])
    assert not torch.allclose(la[:, 20:], lb[:, 20:])


def test_doc_masking_isolates_documents():
    """A document packed after another gives the same logits as when run alone.

    Works without resetting positions per document: RoPE scores depend only on relative position.
    """
    model = make_model()
    doc_a = random_tokens(1, 20)
    doc_a[0, 0] = TINY.eot_id
    doc_b = random_tokens(1, 12, low=1) % 50 + 1
    doc_b[0, 0] = TINY.eot_id
    packed, _ = model(torch.cat([doc_a, doc_b], dim=1))
    alone_a, _ = model(doc_a)
    alone_b, _ = model(doc_b)
    torch.testing.assert_close(packed[:, :20], alone_a, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(packed[:, 20:], alone_b, atol=1e-5, rtol=1e-5)

    unmasked = make_model(doc_masking=False)
    packed, _ = unmasked(torch.cat([doc_a, doc_b], dim=1))
    alone_b, _ = unmasked(doc_b)
    assert not torch.allclose(packed[:, 20:], alone_b, atol=1e-3)


def test_naive_attention_matches_sdpa():
    torch.manual_seed(0)
    B, H, H_kv, T, D = 2, 4, 2, 16, 8
    q, k, v = torch.randn(B, H, T, D), torch.randn(B, H_kv, T, D), torch.randn(B, H_kv, T, D)
    idx = random_tokens(B, T)
    idx[:, [0, 5, 11]] = TINY.eot_id
    mask = build_attention_mask(idx, TINY.eot_id, doc_masking=True)
    ref = F.scaled_dot_product_attention(
        q, k.repeat_interleave(H // H_kv, dim=1), v.repeat_interleave(H // H_kv, dim=1), attn_mask=mask
    )
    torch.testing.assert_close(naive_attention(q, k, v, mask), ref, atol=1e-5, rtol=1e-5)


def test_gqa_equals_mha_with_copied_kv_heads():
    """GQA is MHA where groups of query heads share one K/V head: copy each K/V head and outputs match."""
    gqa = make_model()
    mha = make_model(n_kv_heads=TINY.n_heads)
    n_rep, hd = TINY.n_heads // TINY.n_kv_heads, TINY.head_dim
    state = gqa.state_dict()
    for name, w in state.items():
        if name.endswith(("k_proj.weight", "v_proj.weight")):
            heads = w.view(TINY.n_kv_heads, hd, -1)
            state[name] = heads.repeat_interleave(n_rep, dim=0).reshape(-1, w.shape[1])
    mha.load_state_dict(state)
    idx = random_tokens(2, 32)
    torch.testing.assert_close(gqa(idx)[0], mha(idx)[0], atol=1e-5, rtol=1e-5)


def test_qk_norm_flag():
    diff = make_model().num_params() - make_model(qk_norm=False).num_params()
    assert diff == TINY.n_layers * 2 * TINY.head_dim


def test_overfits_single_batch():
    model = make_model().train()
    idx = random_tokens(4, 33)
    x, y = idx[:, :-1], idx[:, 1:]
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    for _ in range(150):
        _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())
    assert loss.item() < 0.5


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs Apple MPS")
def test_mps_matches_cpu():
    model = make_model()
    idx = random_tokens(2, 32)
    idx[:, 10] = TINY.eot_id
    cpu, _ = model(idx)
    mps, _ = model.to("mps")(idx.to("mps"))
    torch.testing.assert_close(mps.cpu(), cpu, atol=1e-4, rtol=1e-4)
