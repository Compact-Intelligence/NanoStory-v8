#!/usr/bin/env python3
"""
NanoStory v8 — BPE Subword Architecture (Scaled Up)
=====================================================
~148K parameter model — the next step beyond v7's 31K.

Key improvements over v7:
- d_model: 32 → 64 (2x representational capacity)
- GRU hidden: 32 → 64 (deeper sequential processing)
- Attention heads: 2 → 4 (richer attention patterns)
- Transformer blocks: 1 → 2 (deeper reasoning)
- SwiGLU FFN: 64 → 128 (more feedforward capacity)
- Embedding: 26% → 11% (even less embedding dominance)

Architecture:
  Token → Embed(256×64) → [GRU×2 h=64] → [Attn 4h + SwiGLU ff=128]×2 → lm_head → Token

Parameter budget:
  Embedding:  16,384 (11.1%)
  GRU:        49,536 (33.4%)
  Transformer: 82,176 (55.4%)
  Total:     ~148,224 params
  Compute:    88.9% (target >65%)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with ALiBi position biases."""

    def __init__(self, d_model, n_head, block_size):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.head_dim = d_model // n_head

        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # ALiBi slopes
        slopes = torch.tensor([2 ** (-8.0 / n_head * (i + 1)) for i in range(n_head)])
        positions_i = torch.arange(block_size).unsqueeze(1)
        positions_j = torch.arange(block_size).unsqueeze(0)
        distances = positions_j - positions_i
        bias = slopes.unsqueeze(1).unsqueeze(2) * distances.unsqueeze(0)
        self.register_buffer("alibi_bias", bias.unsqueeze(0))

        # Causal mask
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size)
        )

    def forward(self, x):
        B, T, C = x.size()

        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(C, dim=2)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        scale = 1.0 / math.sqrt(self.head_dim)
        att = (q @ k.transpose(-2, -1)) * scale
        att = att + self.alibi_bias[:, :, :T, :T]
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.out_proj(y)
        return y


class SwiGLUFFN(nn.Module):
    """SwiGLU Feed-Forward Network."""

    def __init__(self, d_model, ff_dim):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, ff_dim, bias=False)
        self.up_proj = nn.Linear(d_model, ff_dim, bias=False)
        self.down_proj = nn.Linear(ff_dim, d_model, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    """Pre-norm transformer block: RMSNorm → Attn → RMSNorm → SwiGLU."""

    def __init__(self, d_model, n_head, ff_dim, block_size):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head, block_size)
        self.ln2 = RMSNorm(d_model)
        self.ff = SwiGLUFFN(d_model, ff_dim)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class NanoStoryV8(nn.Module):
    """
    NanoStory v8: ~148K parameter BPE subword model.

    Architecture:
        Token → Embed(256×64) → [GRU×2 h=64] → [Attn 4h + SwiGLU]×2 → lm_head → Token
    """

    def __init__(self, vocab_size=256, d_model=64, n_head=4,
                 gru_hidden=64, ff_dim=128, n_gru=2, n_tf_blocks=2, block_size=128):
        super().__init__()
        self.block_size = block_size
        self.d_model = d_model

        # Token embedding (weight-tied with lm_head)
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # GRU backbone
        self.gru1 = nn.GRU(d_model, gru_hidden, batch_first=True)
        self.gru2 = nn.GRU(gru_hidden, gru_hidden, batch_first=True)
        self.gru_proj = nn.Linear(gru_hidden, d_model, bias=False) if gru_hidden != d_model else nn.Identity()

        # Transformer blocks
        self.tf_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_head, ff_dim, block_size)
            for _ in range(n_tf_blocks)
        ])

        # Final norm
        self.ln_f = RMSNorm(d_model)

        # Weight-tied lm_head
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

        self._init_weights()
        self._report_params()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.GRU):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)

    def _report_params(self):
        n_params = sum(p.numel() for p in self.parameters())
        n_learnable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        vocab_size = self.token_embedding.num_embeddings
        embed_p = vocab_size * self.d_model
        gru_p = sum(p.numel() for p in self.gru1.parameters()) + \
                sum(p.numel() for p in self.gru2.parameters())
        tf_p = sum(p.numel() for p in self.tf_blocks.parameters())
        compute_p = gru_p + tf_p
        print(f"NanoStory v8: {n_params:,} total params, {n_learnable:,} learnable")
        print(f"  Embedding: {embed_p:,} ({embed_p/n_params*100:.1f}%)")
        print(f"  GRU blocks: {gru_p:,} ({gru_p/n_params*100:.1f}%)")
        print(f"  Transformer: {tf_p:,} ({tf_p/n_params*100:.1f}%)")
        print(f"  Compute: {compute_p:,} ({compute_p/n_params*100:.1f}%) — target >65%")

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """Separate weight-decay and non-weight-decay parameter groups."""
        decay_params = []
        no_decay_params = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if param.dim() >= 2:
                decay_params.append(param)
            else:
                no_decay_params.append(param)

        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, fused=False)
        return optimizer

    def forward(self, idx, targets=None, pad_id=0):
        B, T = idx.size()
        assert T <= self.block_size, f"Sequence length {T} exceeds block_size {self.block_size}"

        x = self.token_embedding(idx)

        gru_out1, _ = self.gru1(x)
        gru_out2, _ = self.gru2(gru_out1)
        x = self.gru_proj(gru_out2)

        for block in self.tf_blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            mask = (targets != pad_id).float()
            loss_raw = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                reduction='none'
            )
            loss = (loss_raw.view(B, T) * mask).sum() / mask.sum().clamp(min=1)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=50, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)

        return idx


if __name__ == "__main__":
    print("Testing NanoStoryV8 architecture...\n")
    model = NanoStoryV8()

    x = torch.randint(0, 256, (2, 128))
    logits, loss = model(x, targets=x)
    print(f"\nForward pass: logits shape = {logits.shape}, loss = {loss.item():.4f}")

    x_gen = torch.randint(0, 256, (1, 5))
    out = model.generate(x_gen, max_new_tokens=20, temperature=0.8, top_k=40)
    print(f"Generation: input shape {tuple(x_gen.shape)} → output shape {tuple(out.shape)}")

    # Test optimizer
    opt = model.configure_optimizers(weight_decay=0.01, learning_rate=3e-4, betas=(0.9, 0.95), device_type='cpu')
    print(f"Optimizer: {len(opt.param_groups)} param groups")

    print("\nAll tests passed!")
