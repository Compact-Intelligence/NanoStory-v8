#!/usr/bin/env python3
"""
NanoStory v8 Training Script — BPE Subword (Scaled Up)
=======================================================
Trains the v8 model (~148K params) on ~70K stories with BPE subword
tokenization (256 tokens, block_size=128).

Key differences from v7 (31K params):
- 4.8x more parameters (148K vs 31K)
- 2 transformer blocks (vs 1)
- 4 attention heads (vs 2)
- Batch size 24 (smaller due to bigger model)
- 80K max iterations with cosine LR decay
"""

import os
import time
import math
import pickle
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer

from model_v8 import NanoStoryV8

# ─── Configuration ────────────────────────────────────────
CACHE_DIR = "cache_v8"
CHECKPOINT_DIR = "checkpoints_v8"
LOG_FILE = "train_v8.log"

VOCAB_SIZE = 256
D_MODEL = 64
N_HEAD = 4
GRU_HIDDEN = 64
FF_DIM = 128
N_GRU = 2
N_TF_BLOCKS = 2
BLOCK_SIZE = 128

BATCH_SIZE = 24
MAX_ITERS = 80000
LR = 3e-4
MIN_LR = 3e-5
WARMUP_ITERS = 500
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
EVAL_INTERVAL = 500
SAMPLE_INTERVAL = 2000
EARLY_STOP_PATIENCE = 8000

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3

DEVICE = "cpu"

SAMPLE_PROMPTS = [
    "a little cat",
    "once upon a time",
    "the big dog",
    "a tiny bird",
    "the little girl",
    "one day a rabbit",
]

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ─── Logging ──────────────────────────────────────────────
log = logging.getLogger("nanostory_v8")
log.setLevel(logging.INFO)
for handler in log.handlers[:]:
    log.removeHandler(handler)
fh = logging.FileHandler(LOG_FILE)
fh.setFormatter(logging.Formatter("%(message)s"))
log.addHandler(fh)
sh = logging.StreamHandler()
sh.setFormatter(logging.Formatter("%(message)s"))
log.addHandler(sh)


# ─── Data loading ─────────────────────────────────────────
def load_data():
    with open(os.path.join(CACHE_DIR, "meta.pkl"), "rb") as f:
        meta = pickle.load(f)
    train_data = np.fromfile(os.path.join(CACHE_DIR, "train.bin"), dtype=np.uint16)
    val_data = np.fromfile(os.path.join(CACHE_DIR, "val.bin"), dtype=np.uint16)
    train_data = train_data.reshape(-1, BLOCK_SIZE)
    val_data = val_data.reshape(-1, BLOCK_SIZE)
    return train_data, val_data, meta


def get_batch(data, batch_size):
    ix = np.random.randint(0, len(data), size=batch_size)
    x = torch.tensor(data[ix], dtype=torch.long, device=DEVICE)
    targets = torch.cat(
        [x[:, 1:], torch.full((batch_size, 1), PAD_ID, dtype=torch.long, device=DEVICE)],
        dim=1,
    )
    return x, targets


# ─── Decode helpers ────────────────────────────────────────
@torch.no_grad()
def decode_tokens(tokenizer, token_ids):
    """Decode BPE token IDs back to text, handling Metaspace Ġ prefix."""
    vocab = tokenizer.get_vocab()
    id_to_token = {v: k for k, v in vocab.items()}
    tokens = [id_to_token.get(i, "<UNK>") for i in token_ids]
    text = "".join(t.replace("\u0120", " ") for t in tokens).strip()
    return text


@torch.no_grad()
def generate_samples(model, tokenizer, max_new_tokens=60):
    """Generate sample outputs for qualitative inspection."""
    model.eval()
    samples = []
    for prompt_text in SAMPLE_PROMPTS:
        encoding = tokenizer.encode(prompt_text)
        prompt_ids = [BOS_ID] + encoding.ids
        prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=DEVICE)

        # Greedy
        greedy_out = model.generate(prompt_tensor, max_new_tokens=max_new_tokens, temperature=0.01, top_k=1)
        greedy_ids = greedy_out[0].tolist()
        if EOS_ID in greedy_ids:
            greedy_ids = greedy_ids[:greedy_ids.index(EOS_ID)]
        if BOS_ID in greedy_ids:
            greedy_ids = greedy_ids[greedy_ids.index(BOS_ID) + 1:]
        greedy_text = decode_tokens(tokenizer, greedy_ids)

        # Sampled
        sampled_out = model.generate(prompt_tensor, max_new_tokens=max_new_tokens, temperature=0.8, top_k=40)
        sampled_ids = sampled_out[0].tolist()
        if EOS_ID in sampled_ids:
            sampled_ids = sampled_ids[:sampled_ids.index(EOS_ID)]
        if BOS_ID in sampled_ids:
            sampled_ids = sampled_ids[sampled_ids.index(BOS_ID) + 1:]
        sampled_text = decode_tokens(tokenizer, sampled_ids)

        samples.append((prompt_text, greedy_text, sampled_text))
    model.train()
    return samples


# ─── Learning rate schedule ────────────────────────────────
def get_lr(it):
    if it < WARMUP_ITERS:
        return LR * (it + 1) / WARMUP_ITERS
    if it > MAX_ITERS:
        return MIN_LR
    decay_ratio = (it - WARMUP_ITERS) / (MAX_ITERS - WARMUP_ITERS)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return MIN_LR + coeff * (LR - MIN_LR)


# ─── Main ─────────────────────────────────────────────────
if __name__ == "__main__":
    # Load tokenizer
    log.info("Loading BPE tokenizer...")
    tokenizer = Tokenizer.from_file(os.path.join(CACHE_DIR, "bpe_tokenizer.json"))
    log.info(f"Tokenizer loaded: {tokenizer.get_vocab_size()} tokens")

    # Load data
    log.info("Loading training data...")
    train_data, val_data, meta = load_data()
    log.info(f"Train: {len(train_data):,} sequences, Val: {len(val_data):,} sequences")

    # Initialize model
    log.info(f"\n{'='*60}")
    log.info("Initializing NanoStoryV8 model...")
    log.info(f"{'='*60}")
    model = NanoStoryV8(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        n_head=N_HEAD,
        gru_hidden=GRU_HIDDEN,
        ff_dim=FF_DIM,
        n_gru=N_GRU,
        n_tf_blocks=N_TF_BLOCKS,
        block_size=BLOCK_SIZE,
    )
    model.to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Total learnable parameters: {n_params:,}")

    # Optimizer
    optimizer = model.configure_optimizers(
        weight_decay=WEIGHT_DECAY,
        learning_rate=LR,
        betas=(0.9, 0.95),
        device_type="cpu",
    )

    log.info(f"\n{'='*60}")
    log.info("STARTING TRAINING")
    log.info(f"{'='*60}")
    log.info(f"Max iterations: {MAX_ITERS:,}")
    log.info(f"Batch size: {BATCH_SIZE}")
    log.info(f"Learning rate: {LR} -> {MIN_LR} (cosine)")
    log.info(f"Warmup: {WARMUP_ITERS} iters")
    log.info(f"Early stopping patience: {EARLY_STOP_PATIENCE:,}")
    log.info(f"Eval every {EVAL_INTERVAL}, samples every {SAMPLE_INTERVAL}")
    log.info("")

    best_val_loss = float("inf")
    early_stop_counter = 0
    iter_num = 0
    t0 = time.time()
    tok_per_sec = 0

    while iter_num < MAX_ITERS:
        # Learning rate
        lr = get_lr(iter_num)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Get batch
        xb, yb = get_batch(train_data, BATCH_SIZE)

        # Forward
        logits, loss = model(xb, targets=yb, pad_id=PAD_ID)

        # Backward
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        # Logging
        if iter_num % 100 == 0:
            t1 = time.time()
            elapsed = t1 - t0
            tok_per_sec = BATCH_SIZE * BLOCK_SIZE * 100 / max(elapsed, 0.001)
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
            log.info(f"iter {iter_num} | loss {loss.item():.4f} | lr {lr:.6f} | grad {grad_norm:.4f} | tok/s {tok_per_sec:.0f}")
            t0 = time.time()

        # Evaluation
        if iter_num > 0 and iter_num % EVAL_INTERVAL == 0:
            model.eval()
            val_losses = []
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for _ in range(min(20, len(val_data) // BATCH_SIZE)):
                    xb_v, yb_v = get_batch(val_data, BATCH_SIZE)
                    logits_v, loss_v = model(xb_v, targets=yb_v, pad_id=PAD_ID)
                    val_losses.append(loss_v.item())
                    preds = logits_v.argmax(dim=-1)
                    mask = yb_v != PAD_ID
                    val_correct += (preds[mask] == yb_v[mask]).sum().item()
                    val_total += mask.sum().item()

            avg_val_loss = np.mean(val_losses)
            val_acc = val_correct / max(val_total, 1) * 100

            train_losses = []
            with torch.no_grad():
                for _ in range(10):
                    xb_t, yb_t = get_batch(train_data, BATCH_SIZE)
                    _, loss_t = model(xb_t, targets=yb_t, pad_id=PAD_ID)
                    train_losses.append(loss_t.item())
            avg_train_loss = np.mean(train_losses)

            log.info(f"{'='*50}")
            log.info(f"iter {iter_num} | train_loss {avg_train_loss:.4f} | val_loss {avg_val_loss:.4f} | acc {val_acc:.1f}% | tok/s {tok_per_sec:.0f}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                early_stop_counter = 0
                torch.save({
                    "iter": iter_num,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_loss": avg_val_loss,
                    "val_acc": val_acc,
                }, os.path.join(CHECKPOINT_DIR, "best_model.pt"))
                log.info(f" ** New best val loss: {avg_val_loss:.4f} (acc={val_acc:.1f}%) -> saved checkpoints_v8/best_model.pt")
            else:
                early_stop_counter += EVAL_INTERVAL
                if early_stop_counter >= EARLY_STOP_PATIENCE:
                    log.info(f"\nEARLY STOPPING at iter {iter_num}")
                    log.info(f"Best val loss: {best_val_loss:.4f}")
                    break

            model.train()

        # Sample generation
        if iter_num > 0 and iter_num % SAMPLE_INTERVAL == 0:
            log.info(f"\n{'='*50}")
            log.info("SAMPLE GENERATION")
            log.info(f"{'='*50}")
            samples = generate_samples(model, tokenizer)
            for prompt, greedy, sampled in samples:
                log.info(f"\n  Prompt: {prompt}")
                log.info(f"  Greedy: {greedy}")
                log.info(f"  Sample: {sampled}")
            log.info("")

        iter_num += 1

    # Final summary
    log.info(f"\n{'='*60}")
    log.info("TRAINING COMPLETE")
    log.info(f"{'='*60}")
    log.info(f"Final iteration: {iter_num}")
    log.info(f"Best val loss: {best_val_loss:.4f}")

    ckpt = torch.load(os.path.join(CHECKPOINT_DIR, "best_model.pt"), map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    log.info("\nFinal samples from best model:")
    samples = generate_samples(model, tokenizer)
    for prompt, greedy, sampled in samples:
        log.info(f"\n  Prompt: {prompt}")
        log.info(f"  Greedy: {greedy}")
        log.info(f"  Sample: {sampled}")
