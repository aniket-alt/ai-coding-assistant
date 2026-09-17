"""
Project 02 -- a small decoder-only transformer, written from scratch.

The three Llama-style components are switchable so that each one's value can
be measured rather than assumed:

    position  "rope"    rotary position embeddings (Su et al., 2021)
              "learned" a learned absolute position table (GPT-2)
    mlp       "swiglu"  gated feed-forward (Shazeer, 2020)
              "gelu"    the classic two-layer GELU feed-forward
    norm      "rms"     RMSNorm (Zhang & Sennrich, 2019)
              "layer"   LayerNorm

SwiGLU's hidden width is two-thirds of GELU's so the two variants have the same
parameter count, and an ablation compares like with like.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab_size: int = 4096
    context: int = 256
    d_model: int = 320
    n_layers: int = 6
    n_heads: int = 8
    position: str = "rope"
    mlp: str = "swiglu"
    norm: str = "rms"
    dropout: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def norm(cfg: Config) -> nn.Module:
    return RMSNorm(cfg.d_model) if cfg.norm == "rms" else nn.LayerNorm(cfg.d_model)


def rope_tables(head_dim: int, length: int, base: float = 10000.0, device=None) -> tuple[torch.Tensor, torch.Tensor]:
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    angles = torch.outer(torch.arange(length, device=device).float(), inv)
    return angles.cos(), angles.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate each (even, odd) pair of dimensions by its position's angle. x: (B, H, T, D)."""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    cos, sin = cos[None, None], sin[None, None]
    out = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
    return out.flatten(-2).type_as(x)


class Attention(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x, rope, start: int = 0, cache: dict | None = None, return_weights: bool = False):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=-1)
        q, k, v = (t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2) for t in (q, k, v))
        if rope is not None:
            cos, sin = rope
            q = apply_rope(q, cos[start:start + T], sin[start:start + T])
            k = apply_rope(k, cos[start:start + T], sin[start:start + T])
        if cache is not None:
            if "k" in cache:
                k = torch.cat([cache["k"], k], dim=2)
                v = torch.cat([cache["v"], v], dim=2)
            cache["k"], cache["v"] = k, v
        weights = None
        if return_weights:
            # The explicit path, used only to draw attention maps.
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            mask = torch.ones(T, k.shape[2], dtype=torch.bool, device=x.device).tril(k.shape[2] - T)
            weights = scores.masked_fill(~mask, float("-inf")).softmax(-1)
            y = weights @ v
        else:
            # With a cache the query is only the newest positions, which may see every key.
            y = F.scaled_dot_product_attention(q, k, v, is_causal=cache is None or T > 1)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.out(y), weights


class FeedForward(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.kind = cfg.mlp
        if cfg.mlp == "swiglu":
            hidden = int(2 * 4 * cfg.d_model / 3)
            self.gate = nn.Linear(cfg.d_model, hidden, bias=False)
            self.up = nn.Linear(cfg.d_model, hidden, bias=False)
            self.down = nn.Linear(hidden, cfg.d_model, bias=False)
        else:
            self.up = nn.Linear(cfg.d_model, 4 * cfg.d_model, bias=False)
            self.down = nn.Linear(4 * cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x):
        if self.kind == "swiglu":
            return self.down(F.silu(self.gate(x)) * self.up(x))
        return self.down(F.gelu(self.up(x)))


class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.n1, self.attn, self.n2, self.ff = norm(cfg), Attention(cfg), norm(cfg), FeedForward(cfg)

    def forward(self, x, rope, start=0, cache=None, return_weights=False):
        a, w = self.attn(self.n1(x), rope, start, cache, return_weights)
        x = x + a
        return x + self.ff(self.n2(x)), w


class NanoLM(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = nn.Embedding(cfg.context, cfg.d_model) if cfg.position == "learned" else None
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.final_norm = norm(cfg)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.embed.weight  # tied input and output embeddings
        cos, sin = rope_tables(cfg.d_model // cfg.n_heads, cfg.context)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init)

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None, start: int = 0, caches: list | None = None, return_weights: bool = False):
        B, T = idx.shape
        x = self.embed(idx)
        if self.pos is not None:
            x = x + self.pos(torch.arange(start, start + T, device=idx.device))
        rope = (self.rope_cos, self.rope_sin) if self.cfg.position == "rope" else None
        maps = []
        for i, block in enumerate(self.blocks):
            x, w = block(x, rope, start, caches[i] if caches is not None else None, return_weights)
            if return_weights:
                maps.append(w)
        logits = self.head(self.final_norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss, maps

    @torch.no_grad()
    def generate(self, prompt: list[int], max_new: int, temperature: float = 0.8, top_p: float = 0.9,
                 repetition_penalty: float = 1.1, use_cache: bool = True, stop_id: int | None = None,
                 seed: int | None = None) -> list[int]:
        """
        Sample a continuation. With `use_cache`, each step feeds only the newest
        token and reuses stored keys and values; without it, the whole sequence
        is recomputed every step. Given the same seed, both return the same tokens.
        """
        device = self.embed.weight.device
        gen = torch.Generator(device=device)
        if seed is not None:
            gen.manual_seed(seed)
        ids = list(prompt)[-self.cfg.context:]
        caches = [dict() for _ in self.blocks] if use_cache else None
        logits, _, _ = self(torch.tensor([ids], device=device), caches=caches)
        out: list[int] = []
        for step in range(max_new):
            nxt = self._sample(logits[0, -1].float(), ids, temperature, top_p, repetition_penalty, gen)
            out.append(nxt)
            ids.append(nxt)
            if nxt == stop_id or len(ids) >= self.cfg.context:
                break
            if use_cache:
                logits, _, _ = self(torch.tensor([[nxt]], device=device), start=len(ids) - 1, caches=caches)
            else:
                logits, _, _ = self(torch.tensor([ids], device=device))
        return out

    @staticmethod
    def _sample(logits, history, temperature, top_p, penalty, gen) -> int:
        if penalty != 1.0 and history:
            seen = torch.tensor(sorted(set(history)), device=logits.device)
            vals = logits[seen]
            logits[seen] = torch.where(vals > 0, vals / penalty, vals * penalty)
        if temperature <= 0:
            return int(logits.argmax())
        probs = (logits / temperature).softmax(-1)
        sorted_p, order = probs.sort(descending=True)
        keep = sorted_p.cumsum(-1) - sorted_p < top_p
        sorted_p = sorted_p * keep
        choice = torch.multinomial(sorted_p / sorted_p.sum(), 1, generator=gen)
        return int(order[choice])
